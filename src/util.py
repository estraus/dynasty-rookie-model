"""Shared config + parquet-cache helpers. Everything reads config from here."""
from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
import pyarrow as pa
import yaml

ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def raw_dir() -> Path:
    d = ROOT / config()["paths"]["raw"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached(name: str, builder, *, refresh: bool = False) -> pd.DataFrame:
    """Return data/raw/<name>.parquet, building + caching it on first call.

    `builder` is a zero-arg callable that returns a DataFrame. Raw pulls are the
    slow, rate-limited, key-gated part; cache once so reruns are free.
    """
    path = raw_dir() / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = builder()
    _to_parquet(df, path)
    return df


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet, stringifying mixed-type object columns pyarrow rejects.

    Some nflverse columns (e.g. jersey_number) mix floats and strings in one
    object column; pyarrow can't infer a type. Cast object columns to string
    on failure and retry rather than losing the whole table.
    """
    try:
        df.to_parquet(path, index=False)
    except (pa.lib.ArrowTypeError, pa.lib.ArrowInvalid):
        obj_cols = df.select_dtypes(include="object").columns
        df = df.copy()
        for c in obj_cols:
            df[c] = df[c].map(lambda v: v if v is None else str(v))
        df.to_parquet(path, index=False)
