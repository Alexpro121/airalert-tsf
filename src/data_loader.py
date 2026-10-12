from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import KYIV_TZ

OFFSET_RE = r"\d{2}:\d{2}.*(?:[+-]\d{2}:?\d{2}|Z)$"


def parse_wall_times(s: pd.Series) -> pd.Series:
    """One convention project-wide: offset honored, naive = Europe/Kyiv (first occurrence)."""
    raw = s.astype(str)
    has_offset = raw.str.contains(OFFSET_RE)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    if has_offset.any():
        out[has_offset] = pd.to_datetime(raw[has_offset], utc=True, errors="coerce", format="mixed")
    if (~has_offset).any():
        naive = pd.to_datetime(raw[~has_offset], errors="coerce", format="mixed")
        n_nat = int(naive.isna().sum())
        if n_nat:
            logger.warning(f"{n_nat} naive timestamps unparsable")
        out[~has_offset] = naive.dt.tz_localize(
            KYIV_TZ, ambiguous=True, nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
    n_nat = int(out.isna().sum())
    if n_nat:
        logger.warning(f"{n_nat} timestamps NaT after parsing")
    return out


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    """One convention: strings with an offset are honored; naive strings are Kyiv-local."""
    before = len(df)
    raw_end = df["end_time"].astype(str)
    for col in ("start_time", "end_time"):
        df[col] = parse_wall_times(df[col])
    bad = df["end_time"] < df["start_time"]
    if bad.any():
        naive_mask = ~raw_end.loc[bad].str.contains(OFFSET_RE)
        fixed = df.loc[bad, "end_time"].copy()
        if naive_mask.any():
            fixed[naive_mask] = _refix_fold_end(raw_end.loc[bad][naive_mask])
        df.loc[bad, "end_time"] = fixed
    df = df.dropna(subset=["start_time", "end_time"])
    df = df[df["end_time"] > df["start_time"]].copy()
    dropped = before - len(df)
    if dropped:
        logger.warning(f"dropped {dropped} rows with unparsable/negative-duration times")
    return df


def _refix_fold_end(raw_strings: pd.Series) -> pd.Series:
    """Re-resolve naive end-time strings with the second fold occurrence."""
    naive = pd.to_datetime(raw_strings, errors="coerce")
    fixed = naive.dt.tz_localize(KYIV_TZ, ambiguous=False, nonexistent="shift_forward").dt.tz_convert("UTC")
    logger.info(f"fold-fix re-resolved {int(fixed.notna().sum())}/{len(raw_strings)} ends to second occurrence")
    return fixed


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"region": str, "alert_type": str})
    if "alert_type" not in df.columns or "region" not in df.columns:
        raise ValueError(f"{path} must contain 'region' and 'alert_type' columns")
    df = df[df["alert_type"] == "air_raid"].copy()
    df["region"] = normalize_regions(df["region"])
    df = _parse_times(df)
    return df[["region", "start_time", "end_time", "alert_type"]]


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_meta(path: str | Path, source: str, extra: dict | None = None) -> None:
    import datetime
    import json

    meta = {
        "source": source,
        "sha256": sha256_file(path),
        "fetch_date": datetime.datetime.now(datetime.UTC).isoformat(),
        "pinned": False,
        **(extra or {}),
    }
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2, default=str))


def read_source(path: str | Path) -> str:
    import json

    mp = Path(str(path) + ".meta.json")
    if not mp.exists():
        return "unknown"
    try:
        meta = json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning(f"unreadable meta {mp} -> unknown")
        return "unknown"
    if "sha256" not in meta:
        logger.warning(f"meta {mp} has no sha256 (predates hash rule) -> unknown")
        return "unknown"
    if meta["sha256"] != sha256_file(path):
        logger.warning(f"sha256 mismatch for {path} -> unknown")
        return "unknown"
    if meta.get("source") == "synthetic":
        return "synthetic"
    if meta.get("created_by") != "airalert-tsf.fetch_real_alerts":
        logger.warning(f"meta {mp} not created by fetch_real_alerts -> unknown")
        return "unknown"
    return meta.get("source", "unknown")


def validate_regions(df: pd.DataFrame, required: list[str], min_events: int = 100) -> None:
    counts = df["region"].value_counts()
    absent = [r for r in required if counts.get(r, 0) == 0]
    if absent:
        raise ValueError(f"regions with zero events: {absent} (counts={counts.to_dict()})")
    thin = [r for r in required if counts.get(r, 0) < min_events]
    if thin:
        logger.warning(f"regions below {min_events} events: {thin} (counts={counts.to_dict()})")


VOLUNTEER_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset"
    "/main/datasets/volunteer_data_en.csv"
)
OFFICIAL_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset"
    "/main/datasets/official_data_en.csv"
)

SYNTH_END = pd.Timestamp("2026-10-10 00:00:00", tz="UTC")

_REGION_MAP = {
    "kyiv city": "Kyiv",
    "kyivska oblast": "Kyiv_Oblast",
    "kyiv oblast": "Kyiv_Oblast",
    "kyiv_oblast": "Kyiv_Oblast",
    "kyiv": "Kyiv",
    "chernihivska oblast": "Chernihiv",
    "chernihiv oblast": "Chernihiv",
    "chernihiv": "Chernihiv",
    "sumska oblast": "Sumy",
    "sumy oblast": "Sumy",
    "sumy": "Sumy",
    "poltavska oblast": "Poltava",
    "poltava oblast": "Poltava",
    "poltava": "Poltava",
    "cherkaska oblast": "Cherkasy_Oblast",
    "cherkasy oblast": "Cherkasy_Oblast",
    "cherkasy": "Cherkasy_Oblast",
    "chernivetska oblast": "Chernivtsi_Oblast",
    "chernivtsi oblast": "Chernivtsi_Oblast",
    "chernivtsi": "Chernivtsi_Oblast",
    "dnipropetrovska oblast": "Dnipro_Oblast",
    "dnipropetrovsk oblast": "Dnipro_Oblast",
    "dnipro": "Dnipro_Oblast",
    "donetska oblast": "Donetsk_Oblast",
    "donetsk oblast": "Donetsk_Oblast",
    "donetsk": "Donetsk_Oblast",
    "ivano-frankivska oblast": "IvanoFrankivsk_Oblast",
    "ivano-frankivsk oblast": "IvanoFrankivsk_Oblast",
    "ivano-frankivsk": "IvanoFrankivsk_Oblast",
    "kharkivska oblast": "Kharkiv_Oblast",
    "kharkiv oblast": "Kharkiv_Oblast",
    "kharkiv": "Kharkiv_Oblast",
    "khersonska oblast": "Kherson_Oblast",
    "kherson oblast": "Kherson_Oblast",
    "kherson": "Kherson_Oblast",
    "khmelnytska oblast": "Khmelnytskyi_Oblast",
    "khmelnytskyi oblast": "Khmelnytskyi_Oblast",
    "khmelnytskyi": "Khmelnytskyi_Oblast",
    "kirovohradska oblast": "Kirovohrad_Oblast",
    "kirovohrad oblast": "Kirovohrad_Oblast",
    "kirovohrad": "Kirovohrad_Oblast",
    "luhanska oblast": "Luhansk_Oblast",
    "luhansk oblast": "Luhansk_Oblast",
    "luhansk": "Luhansk_Oblast",
    "lvivska oblast": "Lviv_Oblast",
    "lviv oblast": "Lviv_Oblast",
    "lviv": "Lviv_Oblast",
    "mykolaivska oblast": "Mykolaiv_Oblast",
    "mykolaiv oblast": "Mykolaiv_Oblast",
    "mykolaiv": "Mykolaiv_Oblast",
    "odeska oblast": "Odesa_Oblast",
    "odesa oblast": "Odesa_Oblast",
    "odesa": "Odesa_Oblast",
    "rivnenska oblast": "Rivne_Oblast",
    "rivne oblast": "Rivne_Oblast",
    "rivne": "Rivne_Oblast",
    "ternopilska oblast": "Ternopil_Oblast",
    "ternopil oblast": "Ternopil_Oblast",
    "ternopil": "Ternopil_Oblast",
    "vinnytska oblast": "Vinnytsia_Oblast",
    "vinnytsia oblast": "Vinnytsia_Oblast",
    "vinnytsia": "Vinnytsia_Oblast",
    "volynska oblast": "Volyn_Oblast",
    "volyn oblast": "Volyn_Oblast",
    "volyn": "Volyn_Oblast",
    "zakarpatska oblast": "Zakarpattia_Oblast",
    "zakarpattia oblast": "Zakarpattia_Oblast",
    "zakarpattia": "Zakarpattia_Oblast",
    "zaporizka oblast": "Zaporizhzhia_Oblast",
    "zaporizhzhia oblast": "Zaporizhzhia_Oblast",
    "zaporizhzhia": "Zaporizhzhia_Oblast",
    "zhytomyrska oblast": "Zhytomyr_Oblast",
    "zhytomyr oblast": "Zhytomyr_Oblast",
    "zhytomyr_oblast": "Zhytomyr_Oblast",
    "zhytomyr": "Zhytomyr_Oblast",
    "kharkiv_oblast": "Kharkiv_Oblast",
    "cherkasy_oblast": "Cherkasy_Oblast",
    "vinnytsia_oblast": "Vinnytsia_Oblast",
    "dnipro_oblast": "Dnipro_Oblast",
    "donetsk_oblast": "Donetsk_Oblast",
    "kherson_oblast": "Kherson_Oblast",
    "luhansk_oblast": "Luhansk_Oblast",
    "odesa_oblast": "Odesa_Oblast",
    "zaporizhzhia_oblast": "Zaporizhzhia_Oblast",
}


def normalize_region(name: str) -> str:
    key = str(name).strip().lower()
    if key in _REGION_MAP:
        return _REGION_MAP[key]
    logger.warning(f"unknown region name {name!r}, kept as-is")
    return str(name).strip()


def normalize_regions(s: pd.Series) -> pd.Series:
    """Vectorized: map unique values once, warn once with the full list."""
    stripped = s.astype(str).str.strip()
    keyed = stripped.str.lower()
    uniq = keyed.unique()
    unknown = sorted(u for u in uniq if u not in _REGION_MAP)
    if unknown:
        logger.warning(f"unknown region names kept as-is: {unknown}")
    return keyed.map(_REGION_MAP).fillna(stripped)


def _download(url: str, timeout: int, retries: int = 3) -> bytes:
    import time
    import urllib.error
    import urllib.request

    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "airalert-tsf"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    raise urllib.error.HTTPError(url, resp.status, f"HTTP {resp.status}", resp.headers, None)
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            logger.warning(f"download attempt {attempt + 1}/{retries} failed ({e})")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url}") from last


def fetch_real_alerts(
    path: str | Path, timeout: int = 60, required: list[str] | None = None
) -> pd.DataFrame:
    """Volunteer-дамп Vadimkin (oblast-рівень, з 2022-02-25), fallback — official-дамп.

    Writes to path only after validation; keeps a sha snapshot alongside.
    """
    import io
    import urllib.error

    try:
        raw = pd.read_csv(io.BytesIO(_download(VOLUNTEER_URL, timeout)))
        raw = raw.rename(columns={"started_at": "start_time", "finished_at": "end_time"})
        df = raw[["region", "start_time", "end_time"]].copy()
        df["alert_type"] = "air_raid"
        url = VOLUNTEER_URL
        logger.info(f"volunteer dump rows={len(df)}")
    except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
        logger.warning(f"volunteer download failed ({e}) -> official dump")
        off = pd.read_csv(io.BytesIO(_download(OFFICIAL_URL, timeout)))
        off = off[off["level"] == "oblast"]
        df = pd.DataFrame(
            {
                "region": off["oblast"],
                "start_time": off["started_at"],
                "end_time": off["finished_at"],
                "alert_type": "air_raid",
            }
        )
        url = OFFICIAL_URL
    df["region"] = normalize_regions(df["region"])
    df = _parse_times(df)
    df = df[df["end_time"] > df["start_time"]].copy()
    if required:
        validate_regions(df, required, min_events=1)
    if df.empty:
        raise ValueError("fetched dataset is empty after parsing")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)
    write_meta(tmp, "real", {"url": url, "note": "main branch, not pinned",
                             "dataset_commit": resolve_dataset_commit(),
                             "created_by": "airalert-tsf.fetch_real_alerts"})
    tmp.replace(path)
    Path(str(tmp) + ".meta.json").replace(str(path) + ".meta.json")
    _snapshot(path)
    logger.info(f"real rows={len(df)} -> {path}")
    return df[["region", "start_time", "end_time", "alert_type"]]


def resolve_dataset_commit(path: str = "datasets/volunteer_data_en.csv") -> str | None:
    import json
    import urllib.request

    try:
        url = ("https://api.github.com/repos/Vadimkin/ukrainian-air-raid-sirens-dataset"
               f"/commits?path={path}&per_page=1")
        req = urllib.request.Request(url, headers={"User-Agent": "airalert-tsf"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())[0]["sha"]
    except Exception as e:  # noqa: BLE001 - offline is fine
        logger.warning(f"commit resolve failed ({e})")
        return None


def _snapshot(path: str | Path, max_snapshots: int = 5) -> None:
    digest = sha256_file(path)[:8]
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    snap = Path(path).parent / f"{Path(path).stem}_{stamp}_{digest}.csv"
    if snap.exists():
        return
    import shutil

    shutil.copy(path, snap)
    logger.info(f"snapshot -> {snap}")
    referenced = _referenced_shas()
    snaps = sorted(Path(path).parent.glob(f"{Path(path).stem}_[0-9]*_*.csv"))
    for old in snaps[:-max_snapshots] if len(snaps) > max_snapshots else []:
        if any(old.stem.endswith(s) for s in referenced):
            continue
        old.unlink()
        logger.info(f"removed old snapshot {old}")


def _referenced_shas() -> set[str]:
    """data_sha256 values stored by any reports/*/metrics.json (never delete those snapshots)."""
    import json

    out: set[str] = set()
    for mp in list(Path("reports").glob("*/metrics.json")) + [Path("reports/metrics.json")]:
        try:
            out.add(json.loads(mp.read_text()).get("data_sha256", "")[:8])
        except (OSError, json.JSONDecodeError):
            pass
    return {s for s in out if s}


def generate_synthetic(
    path: str | Path,
    days: int = 400,
    seed: int = 42,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = ["Kyiv", "Chernihiv", "Sumy", "Poltava", "Kyiv_Oblast",
               "Kharkiv_Oblast", "Cherkasy_Oblast", "Zhytomyr_Oblast", "Vinnytsia_Oblast"]
    end = (end or SYNTH_END).floor("h")
    start = end - pd.Timedelta(days=days)
    rows: list[dict] = []
    for r in regions:
        t = start
        while t < end:
            hour = t.tz_convert(KYIV_TZ).hour
            night_boost = 0.10 if (hour >= 23 or hour < 6) else 0.02
            wave = 0.06 if r in ("Chernihiv", "Sumy") else 0.0
            if rng.random() < night_boost + wave:
                dur_min = int(rng.integers(20, 300))
                s = t + pd.Timedelta(minutes=int(rng.integers(0, 60)))
                rows.append(
                    {
                        "region": r,
                        "start_time": s,
                        "end_time": s + pd.Timedelta(minutes=dur_min),
                        "alert_type": "air_raid",
                    }
                )
                if r in ("Chernihiv", "Sumy") and rng.random() < 0.55:
                    lag_min = int(rng.integers(30, 150))
                    ks = s + pd.Timedelta(minutes=lag_min)
                    rows.append(
                        {
                            "region": "Kyiv",
                            "start_time": ks,
                            "end_time": ks + pd.Timedelta(minutes=dur_min),
                            "alert_type": "air_raid",
                        }
                    )
                t += pd.Timedelta(minutes=dur_min + int(rng.integers(120, 1800)))
            else:
                t += pd.Timedelta(hours=1)
    cols = ["region", "start_time", "end_time", "alert_type"]
    if not rows:
        raise ValueError(f"synthetic generator produced 0 rows (days={days}, seed={seed})")
    df = pd.DataFrame(rows, columns=cols).sort_values("start_time").reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    write_meta(path, "synthetic", {"seed": seed, "days": days})
    logger.info(f"synthetic rows={len(df)} -> {path}")
    return df
