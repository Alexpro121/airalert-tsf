import re
import unicodedata
from pathlib import Path

import pandas as pd
from loguru import logger

_CONFUSABLES = str.maketrans({
    "i": "і", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
    "a": "а", "k": "к", "m": "м", "t": "т", "b": "в", "h": "н",
})


def normalize_text(t: str) -> str:
    return unicodedata.normalize("NFKC", t).lower()

THREAT_KEYWORDS = (
    "міг-31", "миг-31", "mig-31", "ту-95", "tu-95", "кинджал", "kinzhal",
    "шахед", "shahed", "герань", "бпла", "uav", "дрон", "drone",
    "балістик", "балістич", "ballistic", "зліт", "takeoff", "ракет",
    "missile", "винищувач",
)

ALL_CLEAR_KEYWORDS = ("відбій", "отбой")

KEYWORDS = THREAT_KEYWORDS


def _word_pattern(keywords: tuple[str, ...]) -> re.Pattern:
    """Left boundary + stem prefix: Ukrainian inflections match, mid-word hits do not."""
    return re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, keywords)) + r")")


def load_events(path: str | Path) -> pd.DataFrame:
    from src.data_loader import parse_wall_times

    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "text"])
    df = pd.read_csv(p)
    df["timestamp"] = parse_wall_times(df["timestamp"])
    df = df[df["timestamp"].notna()]
    return df[["timestamp", "text"]]


def _clauses(t: str) -> list[str]:
    return [c.strip() for c in re.split(r"[.!?;\n—–:]+", t) if c.strip()]


def build_hourly_flags(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    keywords: tuple[str, ...] = THREAT_KEYWORDS,
    exclude_keywords: tuple[str, ...] = (),
) -> pd.Series:
    """Flag hours with keyword hits, scored per clause.

    A message like 'Vidbiy in Kyiv, new Shahed group over Chernihiv' counts as
    threat: the all-clear clause does not cancel the threat clause.
    """
    flags = pd.Series(0, index=index, dtype="int8")
    if events.empty:
        return flags
    if events["timestamp"].dt.tz is None or index.tz is None:
        raise ValueError("events and grid index must both be tz-aware")
    if str(events["timestamp"].dt.tz) != str(index.tz):
        raise ValueError(f"tz mismatch: events={events['timestamp'].dt.tz} grid={index.tz}")
    pat = _word_pattern(keywords)
    excl = _word_pattern(exclude_keywords) if exclude_keywords else None
    text = events["text"].fillna("").apply(normalize_text)
    cyr = text.apply(lambda t: t.translate(_CONFUSABLES))

    def _row_hit(t: str, c: str) -> bool:
        for clause in _clauses(t):
            if pat.search(clause) and not (excl and excl.search(clause)):
                return True
        for clause in _clauses(c):
            if pat.search(clause) and not (excl and excl.search(clause)):
                return True
        return False

    hit = pd.Series([_row_hit(t, c) for t, c in zip(text, cyr)], index=events.index)
    hours = events.loc[hit, "timestamp"].dt.floor("h")
    in_grid = flags.index.intersection(hours)
    flags.loc[in_grid] = 1
    n_out = int((~hours.isin(flags.index)).sum())
    logger.info(
        f"text flags: {int(hit.sum())}/{len(events)} msgs, {len(in_grid.unique())} grid hours"
        + (f", {n_out} msgs outside grid" if n_out else "")
    )
    return flags
