import json
from pathlib import Path

from src.data_loader import read_source, sha256_file, write_meta


def _csv(p: Path) -> None:
    p.write_text("region,start_time,end_time,alert_type\nKyiv,2024-01-01 00:10:00,2024-01-01 00:20:00,air_raid\n")


def test_meta_roundtrip_verifies_hash(tmp_path):
    p = tmp_path / "a.csv"
    _csv(p)
    write_meta(p, "real", {"url": "http://example.invalid/x.csv",
                           "created_by": "airalert-tsf.fetch_real_alerts"})
    assert read_source(p) == "real"
    meta = json.loads(Path(str(p) + ".meta.json").read_text())
    assert sha256_file(p) == meta["sha256"]


def test_wrong_hash_means_unknown(tmp_path):
    p = tmp_path / "a.csv"
    _csv(p)
    write_meta(p, "real", {})
    meta = json.loads(Path(str(p) + ".meta.json").read_text())
    meta["sha256"] = "0" * 64
    Path(str(p) + ".meta.json").write_text(json.dumps(meta))
    assert read_source(p) == "unknown"


def test_legacy_meta_without_hash_is_unknown(tmp_path):
    p = tmp_path / "a.csv"
    _csv(p)
    Path(str(p) + ".meta.json").write_text(json.dumps({"source": "real"}))
    assert read_source(p) == "unknown"


def test_synthetic_empty_days_raises(tmp_path):
    from src.data_loader import generate_synthetic

    try:
        generate_synthetic(tmp_path / "s.csv", days=0, seed=42)
    except ValueError as e:
        assert "0 rows" in str(e)
    else:
        raise AssertionError("expected ValueError")
