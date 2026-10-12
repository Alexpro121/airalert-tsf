"""AirAlert-TSF: prognoz cherez brauzer. Zapusk: uv run python app.py"""

import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

SURS_ROOT = Path(__file__).parent
SURS_KYIV = ZoneInfo("Europe/Kyiv")
SURS_UTC = ZoneInfo("UTC")

surs_model = None
surs_fichy = []
surs_porig = 0.5
surs_X = None
surs_sitka = None
surs_meta = {}
surs_html = ""
surs_min_chas = None
surs_max_chas = None
surs_cfg = None
surs_live_sha = ""
surs_mediany = {}
surs_sync_lock = threading.Lock()
surs_sync_yde = False


def surs_ryzyk(surs_proba: np.ndarray) -> np.ndarray:
    surs_proba = np.asarray(surs_proba)
    surs_klasy = list(getattr(surs_model, "classes_", [0, 1]))
    if len(surs_klasy) == 1:
        if surs_klasy[0] == 1:
            return np.full(surs_proba.shape[0], 1.0)
        return np.full(surs_proba.shape[0], 0.0)
    if surs_proba.ndim == 1:
        return surs_proba.astype(float)
    surs_idx = list(surs_klasy).index(1)
    return surs_proba[:, surs_idx].astype(float)


def surs_pobuduvaty() -> tuple:
    from src.data_loader import load_raw, sha256_file
    from src.feature_engineering import make_features
    from src.grid_builder import build_hourly_grid, coverage_mask

    surs_sha2 = sha256_file(SURS_ROOT / surs_cfg.raw_path)[:8]
    surs_df = load_raw(SURS_ROOT / surs_cfg.raw_path)
    surs_regiony = [surs_cfg.target_region] + [n for n in surs_cfg.neighbors if n != surs_cfg.target_region]
    surs_sitka2 = build_hourly_grid(surs_df, surs_regiony, freq=surs_cfg.freq)
    surs_sitka2 = surs_sitka2.where(coverage_mask(surs_df, surs_regiony, surs_sitka2.index))
    if surs_cfg.tail_cutoff_hours:
        surs_sitka2 = surs_sitka2.loc[: surs_sitka2.index.max() - pd.Timedelta(hours=surs_cfg.tail_cutoff_hours)]
    surs_X2, _ = make_features(
        surs_sitka2, surs_cfg.target_region, surs_cfg.neighbors,
        horizon=surs_cfg.horizon_h, lags=surs_cfg.lags, windows=surs_cfg.windows,
    )
    surs_brak = [c for c in surs_fichy if c not in surs_X2.columns]
    if surs_brak:
        raise ValueError(f"model needs {surs_brak}, have only {list(surs_X2.columns)}")
    surs_med2 = {}
    surs_hvist = surs_X2[surs_fichy].iloc[-720:]
    for surs_c in surs_fichy:
        surs_med2[surs_c] = float(surs_hvist[surs_c].median())
    surs_min2 = surs_X2.index.min() + pd.Timedelta(hours=24)
    surs_max2 = surs_X2.index.max()
    return surs_sitka2, surs_X2, surs_min2, surs_max2, surs_med2, surs_sha2


def surs_zastosuvaty(surs_paket: tuple) -> None:
    global surs_X, surs_sitka, surs_min_chas, surs_max_chas, surs_live_sha
    surs_sitka, surs_X, surs_min_chas, surs_max_chas, surs_med_novi, surs_live_sha = surs_paket
    surs_mediany.clear()
    surs_mediany.update(surs_med_novi)


def surs_zavantazhyty() -> None:
    global surs_model, surs_fichy, surs_porig, surs_meta, surs_html, surs_cfg
    from src.config import Settings

    print("завантаження моделі...", flush=True)
    if not (SURS_ROOT / "reports" / "model.pkl").exists():
        raise SystemExit("нема reports/model.pkl — спочатку: uv run python main.py --region Kyiv")
    surs_bandl = joblib.load(SURS_ROOT / "reports" / "model.pkl")
    surs_model = surs_bandl["model"]
    surs_fichy = list(surs_bandl["feature_names"])
    surs_porig = float(surs_bandl.get("threshold", 0.5))
    surs_meta = json.loads((SURS_ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    surs_cfg0 = surs_meta.get("config", {})
    surs_cfg = Settings(
        target_region=surs_meta.get("target", "Kyiv"),
        neighbors=surs_cfg0.get("neighbors", Settings().neighbors),
        horizon_h=surs_meta.get("horizon_h", 3),
        lags=surs_cfg0.get("lags", [1, 2, 3, 6, 24]),
        windows=surs_cfg0.get("windows", [6, 12, 24]),
        test_days=surs_meta.get("test_days", 30),
    )
    surs_zastosuvaty(surs_pobuduvaty())
    surs_html = (SURS_ROOT / "app.html").read_text(encoding="utf-8")
    print(f"готово: {len(surs_X)} годин, {surs_min_chas} → {surs_max_chas}", flush=True)


def surs_sync() -> dict:
    global surs_sync_yde
    with surs_sync_lock:
        if surs_sync_yde:
            raise ValueError("синхронізація вже йде, зачекай")
        surs_sync_yde = True
    try:
        from src.data_loader import fetch_real_alerts

        try:
            surs_df2 = fetch_real_alerts(
                SURS_ROOT / surs_cfg.raw_path,
                required=[surs_cfg.target_region, *surs_cfg.neighbors],
            )
        except Exception as surs_e:  # noqa: BLE001
            raise ValueError(f"не стягнув дамп (нема інтернету?): {surs_e}")
        print("синхронізація: дамп стягнуто, перебудовую сітку...", flush=True)
        surs_zastosuvaty(surs_pobuduvaty())
        print(f"синхронізація: {len(surs_X)} годин, {surs_min_chas} → {surs_max_chas}", flush=True)
        return {
            "rows": len(surs_df2),
            "data_from": surs_min_chas.isoformat(),
            "data_to": surs_max_chas.isoformat(),
            "hours": len(surs_X),
        }
    finally:
        with surs_sync_lock:
            surs_sync_yde = False


def surs_rozibraty_chas(surs_tekst: str) -> pd.Timestamp:
    surs_tekst = surs_tekst.strip().replace("Z", "+00:00")
    try:
        surs_dt = datetime.fromisoformat(surs_tekst)
    except ValueError:
        raise ValueError(f"не дата: {surs_tekst!r} (треба РРРР-ММ-ДДТГГ:ХХ)")
    if surs_dt.tzinfo is None:
        surs_dt = surs_dt.replace(tzinfo=SURS_KYIV)
    surs_dt = surs_dt.astimezone(SURS_UTC).replace(minute=0, second=0, microsecond=0)
    return pd.Timestamp(surs_dt)


def surs_prognoz(surs_ts: pd.Timestamp) -> dict:
    if surs_ts < surs_min_chas or surs_ts > surs_max_chas:
        raise ValueError(
            f"прогноз можливий лише {surs_min_chas} … {surs_max_chas} (UTC); "
            f"свіжіші дані: uv run python main.py --real"
        )
    surs_riadok = surs_X.loc[surs_ts:surs_ts]
    if surs_riadok.empty:
        raise ValueError(f"нема години {surs_ts} в сітці")
    surs_start = time.perf_counter()
    surs_p = surs_ryzyk(surs_model.predict_proba(surs_riadok[surs_fichy]))[0]
    surs_ms = round((time.perf_counter() - surs_start) * 1000, 1)
    surs_ryzyk_v = round(float(surs_p), 4)
    surs_cil = surs_meta.get("target", "Kyiv")
    surs_susidy = []
    for c in surs_fichy:
        if c == "nbr_new_onset":
            continue
        if c.startswith("nbr_") and not c.endswith("_lag2") and not c.endswith("_sum3"):
            surs_v = surs_riadok[c].iloc[0]
            if float(surs_v) == 1.0:
                surs_susidy.append(c[4:])
    surs_god = surs_ts.tz_convert(SURS_KYIV).hour
    return {
        "timestamp_utc": surs_ts.isoformat(),
        "timestamp_kyiv": surs_ts.tz_convert(SURS_KYIV).isoformat(),
        "region": surs_cil,
        "horizon_h": int(surs_meta.get("horizon_h", 3)),
        "risk": surs_ryzyk_v,
        "risk_pct": round(surs_ryzyk_v * 100, 1),
        "threshold": surs_porig,
        "alarm": bool(surs_ryzyk_v >= surs_porig),
        "context": {
            "lag_1": int(float(surs_riadok["lag_1"].iloc[0])) if "lag_1" in surs_riadok else None,
            "kyiv_hour": int(surs_god),
            "is_night": bool(int(float(surs_riadok["is_night"].iloc[0]))) if "is_night" in surs_riadok else None,
            "active_neighbors": surs_susidy,
            "country_ratio": round(float(surs_riadok["country_ratio"].iloc[0]), 3) if "country_ratio" in surs_riadok else None,
            "hours_since_last": float(surs_riadok["hours_since_last"].iloc[0]) if "hours_since_last" in surs_riadok else None,
        },
        "pipeline": {
            "model": type(surs_model).__name__,
            "features": len(surs_fichy),
            "grid_hours": len(surs_sitka),
            "alerts_24h": int(surs_sitka[surs_cil].iloc[-24:].fillna(0).sum()),
            "ms": surs_ms,
        },
    }


def surs_poyasnennya(surs_ts: pd.Timestamp) -> dict:
    surs_riadok = surs_X.loc[surs_ts:surs_ts]
    if surs_riadok.empty:
        raise ValueError(f"нема години {surs_ts} в сітці")
    surs_baza = pd.DataFrame([{c: surs_mediany[c] for c in surs_fichy}])
    surs_baz_ryzyk = round(float(surs_ryzyk(surs_model.predict_proba(surs_baza))[0]), 4)
    surs_full = round(float(surs_ryzyk(surs_model.predict_proba(surs_riadok[surs_fichy]))[0]), 4)
    surs_drayvery = []
    for surs_c in surs_fichy:
        surs_prob = surs_riadok[surs_fichy].copy()
        surs_prob[surs_c] = surs_mediany[surs_c]
        surs_p2 = float(surs_ryzyk(surs_model.predict_proba(surs_prob))[0])
        surs_v = surs_riadok[surs_c].iloc[0]
        surs_drayvery.append({
            "name": surs_c,
            "value": None if pd.isna(surs_v) else round(float(surs_v), 3),
            "median": round(surs_mediany[surs_c], 3),
            "delta": round(surs_full - surs_p2, 4),
        })
    surs_drayvery.sort(key=lambda surs_d: abs(surs_d["delta"]), reverse=True)
    return {"risk": surs_full, "base": surs_baz_ryzyk, "drivers": surs_drayvery[:8]}


def surs_istoria(surs_dniv: int) -> dict:
    surs_dniv = max(1, min(30, surs_dniv))
    surs_kinets = surs_max_chas
    surs_poch = surs_kinets - pd.Timedelta(days=surs_dniv)
    surs_shmat = surs_X[surs_X.index > surs_poch]
    surs_p = surs_ryzyk(surs_model.predict_proba(surs_shmat[surs_fichy]))
    surs_cil = surs_meta.get("target", "Kyiv")
    surs_t, surs_r, surs_f = [], [], []
    for i, surs_ts in enumerate(surs_shmat.index):
        surs_t.append(surs_ts.isoformat())
        surs_r.append(round(float(surs_p[i]), 4))
        surs_v = surs_sitka.loc[surs_ts, surs_cil] if surs_ts in surs_sitka.index else float("nan")
        surs_f.append(None if pd.isna(surs_v) else int(surs_v))
    return {"days": surs_dniv, "threshold": surs_porig, "t": surs_t, "risk": surs_r, "actual": surs_f}


class SursRuky(BaseHTTPRequestHandler):
    server_version = "AirAlert-TSF/1.0"

    def surs_viddaty(self, surs_kod: int, surs_tilo: bytes, surs_typ: str) -> None:
        self.send_response(surs_kod)
        self.send_header("Content-Type", surs_typ)
        self.send_header("Content-Length", str(len(surs_tilo)))
        self.end_headers()
        self.wfile.write(surs_tilo)

    def surs_json(self, surs_kod: int, surs_obj: dict) -> None:
        surs_tilo = json.dumps(surs_obj, ensure_ascii=False).encode()
        self.send_response(surs_kod)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(surs_tilo)))
        self.end_headers()
        self.wfile.write(surs_tilo)

    def do_GET(self) -> None:
        surs_rozbir = urlparse(self.path)
        surs_shliah = surs_rozbir.path
        surs_q = parse_qs(surs_rozbir.query)
        try:
            if surs_shliah in ("/", "/app.html"):
                self.surs_viddaty(200, surs_html.encode(), "text/html; charset=utf-8")
            elif surs_shliah == "/api/status":
                self.surs_json(200, {
                    "target": surs_meta.get("target", "Kyiv"),
                    "backend": surs_meta.get("backend", "?"),
                    "threshold": surs_porig,
                    "horizon_h": int(surs_meta.get("horizon_h", 3)),
                    "data_from": surs_min_chas.isoformat(),
                    "data_to": surs_max_chas.isoformat(),
                    "hours": len(surs_X),
                    "prevalence": surs_meta.get("prevalence", {}),
                    "data_sha256": surs_live_sha,
                    "model_sha256": str(surs_meta.get("data_sha256", ""))[:8],
                    "degenerate_fallback": bool(surs_meta.get("degenerate_fallback", False)),
                })
            elif surs_shliah == "/api/predict":
                surs_ts_raw = (surs_q.get("ts") or [""])[0]
                if not surs_ts_raw:
                    self.surs_json(400, {"error": "дай ts, наприклад ?ts=2026-10-09T20:00"})
                    return
                self.surs_json(200, surs_prognoz(surs_rozibraty_chas(surs_ts_raw)))
            elif surs_shliah == "/api/history":
                try:
                    surs_d = int((surs_q.get("days") or ["3"])[0])
                except ValueError:
                    surs_d = 3
                self.surs_json(200, surs_istoria(surs_d))
            elif surs_shliah == "/api/explain":
                surs_ts_raw = (surs_q.get("ts") or [""])[0]
                if not surs_ts_raw:
                    self.surs_json(400, {"error": "дай ts, наприклад ?ts=2026-10-09T20:00"})
                    return
                self.surs_json(200, surs_poyasnennya(surs_rozibraty_chas(surs_ts_raw)))
            else:
                self.surs_json(404, {"error": "нема такої сторінки"})
        except ValueError as surs_e:
            self.surs_json(400, {"error": str(surs_e)})
        except Exception as surs_e:  # noqa: BLE001
            self.surs_json(500, {"error": f"внутрішня помилка: {surs_e}"})

    def do_POST(self) -> None:
        surs_shliah = urlparse(self.path).path
        try:
            if surs_shliah == "/api/sync":
                self.surs_json(200, surs_sync())
            else:
                self.surs_json(404, {"error": "нема такої сторінки"})
        except ValueError as surs_e:
            self.surs_json(400, {"error": str(surs_e)})
        except Exception as surs_e:  # noqa: BLE001
            self.surs_json(500, {"error": f"внутрішня помилка: {surs_e}"})

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    surs_ap = argparse.ArgumentParser()
    surs_ap.add_argument("--host", default="127.0.0.1")
    surs_ap.add_argument("--port", type=int, default=8000)
    surs_args = surs_ap.parse_args()
    surs_zavantazhyty()
    surs_server = ThreadingHTTPServer((surs_args.host, surs_args.port), SursRuky)
    print(f"відкрий у браузері: http://{surs_args.host}:{surs_args.port}", flush=True)
    try:
        surs_server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
