from pydantic import BaseModel, Field, field_validator

KYIV_TZ = "Europe/Kyiv"

TRANSIT_REGIONS: dict[str, list[str]] = {
    # Threat-transit directions, not geographic adjacency: kept asymmetric on purpose.
    # Kyiv city is NOT a threat source for Sumy, so it is excluded there.
    # Kyiv watches 8 corridors: north (Chernihiv), northeast (Sumy),
    # east (Kharkiv, Poltava), south (Cherkasy), southwest (Vinnytsia),
    # west (Zhytomyr) plus Kyiv_Oblast ring.
    "Kyiv": ["Chernihiv", "Sumy", "Poltava", "Kyiv_Oblast",
             "Kharkiv_Oblast", "Cherkasy_Oblast", "Zhytomyr_Oblast", "Vinnytsia_Oblast"],
    "Chernihiv": ["Kyiv", "Sumy", "Kyiv_Oblast"],
    "Sumy": ["Chernihiv", "Poltava"],
    "Poltava": ["Kyiv", "Sumy", "Chernihiv"],
    "Kyiv_Oblast": ["Kyiv", "Chernihiv", "Sumy", "Poltava"],
    "Kharkiv_Oblast": ["Sumy", "Poltava", "Kyiv"],
    "Cherkasy_Oblast": ["Kyiv", "Poltava", "Vinnytsia_Oblast"],
    "Zhytomyr_Oblast": ["Kyiv", "Kyiv_Oblast"],
    "Vinnytsia_Oblast": ["Kyiv", "Cherkasy_Oblast"],
}


class Settings(BaseModel):
    target_region: str = "Kyiv"
    neighbors: list[str] = Field(
        default_factory=lambda: list(TRANSIT_REGIONS["Kyiv"]), validate_default=True
    )
    horizon_h: int = 3
    freq: str = "1h"
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 6, 24])
    windows: list[int] = Field(default_factory=lambda: [3, 6, 12, 24])
    test_days: int = 30
    n_splits: int = 5
    seed: int = 42
    raw_path: str = "data/raw/alerts.csv"
    synthetic_path: str = "data/raw/alerts_synthetic.csv"
    grid_path: str = "data/processed/grid.csv"
    report_dir: str = "reports"
    airforce_path: str = "data/raw/airforce_messages.csv"
    use_text_signals: bool = False
    tail_cutoff_hours: int = 6
    min_events_per_region: int = 100
    lgbm_params: dict = Field(default_factory=dict)
    threshold_precision_ratio: float = 1.2
    threshold_floor: float = 0.25

    @field_validator("neighbors")
    @classmethod
    def drop_target_from_neighbors(cls, v: list[str], info) -> list[str]:
        target = (info.data or {}).get("target_region", "Kyiv")
        return [n for n in v if n != target]

    @field_validator("lags")
    @classmethod
    def require_lag_1(cls, v: list[int]) -> list[int]:
        if 1 not in v:
            raise ValueError("lags must contain 1 (prev/onset split and baselines need lag_1)")
        return v

    @field_validator("freq")
    @classmethod
    def only_hourly(cls, v: str) -> str:
        if v != "1h":
            raise ValueError(f"only freq='1h' is supported (grid check requires 1h), got {v!r}")
        return v

    @field_validator("tail_cutoff_hours")
    @classmethod
    def non_negative_tail(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"tail_cutoff_hours must be >= 0 (0=auto p99), got {v}")
        return v

    @field_validator("min_events_per_region")
    @classmethod
    def positive_min_events(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_events_per_region must be >= 1, got {v}")
        return v

    @field_validator("threshold_precision_ratio")
    @classmethod
    def positive_ratio(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"threshold_precision_ratio must be > 0, got {v}")
        return v

    @field_validator("threshold_floor")
    @classmethod
    def floor_in_range(cls, v: float) -> float:
        if not 0 <= v < 1:
            raise ValueError(f"threshold_floor must be in [0, 1), got {v}")
        return v

    @field_validator("lgbm_params")
    @classmethod
    def protect_seed(cls, v: dict) -> dict:
        blocked = {"random_state", "deterministic", "verbose"} & set(v)
        if blocked:
            raise ValueError(f"lgbm_params must not override {sorted(blocked)} (from seed)")
        return v

    @field_validator("horizon_h")
    @classmethod
    def positive_horizon(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"horizon_h must be >= 1, got {v}")
        return v

    @field_validator("n_splits")
    @classmethod
    def enough_splits(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"n_splits must be >= 2, got {v}")
        return v

    @field_validator("test_days")
    @classmethod
    def positive_test_days(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"test_days must be > 0, got {v}")
        return v

    @field_validator("windows", "lags")
    @classmethod
    def positive_ints(cls, v: list[int]) -> list[int]:
        if not v or any(x <= 0 for x in v):
            raise ValueError(f"windows/lags must be non-empty positive ints, got {v}")
        return v

    @field_validator("neighbors")
    @classmethod
    def known_neighbors(cls, v: list[str]) -> list[str]:
        known = set(TRANSIT_REGIONS)
        unknown = [n for n in v if n not in known]
        if unknown:
            from loguru import logger

            logger.warning(f"neighbors outside TRANSIT_REGIONS: {unknown} (must exist in data)")
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate regions in neighbors: {v}")
        return v

    def resolved_lgbm_params(self) -> dict:
        p = {
            "objective": "binary",
            "max_depth": 6,
            "learning_rate": 0.02,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "subsample_freq": 1,
            "min_child_samples": 40,
            "n_estimators": 500,
            "random_state": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 4,
            "verbose": -1,
        }
        p.update(self.lgbm_params)
        return p
