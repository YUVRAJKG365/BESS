import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Any


def safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None for NaN/Inf/None."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or (isinstance(value, float) and np.isinf(value)):
        return None
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def safe_int(value: Any) -> Optional[int]:
    """Convert a value to int, returning None for invalid inputs."""
    f = safe_float(value)
    if f is None:
        return None
    return int(f)


def parse_date(value: Any) -> Optional[pd.Timestamp]:
    """Parse a date value, returning None for invalid inputs."""
    if value is None:
        return None
    try:
        return pd.to_datetime(value, errors='coerce')
    except Exception:
        return None


def format_date_display(ts: pd.Timestamp) -> str:
    """Format a timestamp for display."""
    if ts is None:
        return "N/A"
    return ts.strftime('%Y-%m-%d')


def detect_file_type(file_path: str) -> str:
    """Detect file type based on extension."""
    ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
    if ext in ['xlsx', 'xls']:
        return 'excel'
    elif ext == 'csv':
        return 'csv'
    else:
        return 'unknown'


def calculate_dates_range(df: pd.DataFrame, date_col: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Calculate min/max dates from a column."""
    if date_col not in df.columns:
        return None, None
    dates = pd.to_datetime(df[date_col], errors='coerce')
    valid_dates = dates.dropna()
    if len(valid_dates) == 0:
        return None, None
    return valid_dates.min(), valid_dates.max()


def calculate_duration_hours(start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    """Calculate duration in hours between two timestamps."""
    if start is None or end is None:
        return None
    try:
        delta = end - start
        return delta.total_seconds() / 3600.0
    except Exception:
        return None


def dod_from_charge_discharge(charge_pct: float, discharge_pct: float) -> float:
    """Calculate DoD from charge/discharge percentages."""
    return max(0.0, min(100.0, charge_pct + discharge_pct))


def soc_average(df: pd.DataFrame, soc_col: str) -> Optional[float]:
    """Calculate average SOC from a DataFrame column."""
    if soc_col not in df.columns:
        return None
    valid = pd.to_numeric(df[soc_col], errors='coerce').dropna()
    if len(valid) == 0:
        return None
    return float(valid.mean())


def soc_std(df: pd.DataFrame, soc_col: str) -> Optional[float]:
    """Calculate standard deviation of SOC."""
    if soc_col not in df.columns:
        return None
    valid = pd.to_numeric(df[soc_col], errors='coerce').dropna()
    if len(valid) < 2:
        return None
    return float(valid.std())


def temperature_stats(df: pd.DataFrame, temp_col: str) -> Dict[str, Optional[float]]:
    """Calculate temperature statistics."""
    if temp_col not in df.columns:
        return {"min": None, "max": None, "mean": None}
    valid = pd.to_numeric(df[temp_col], errors='coerce').dropna()
    if len(valid) == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
    }


def cycle_count_from_events(df: pd.DataFrame, event_col: str) -> Optional[int]:
    """Count cycles from event markers."""
    if event_col not in df.columns:
        return None
    valid = pd.to_numeric(df[event_col], errors='coerce').dropna()
    # Count upward crossings or event markers
    if len(valid) == 0:
        return None
    # Simple: count non-zero events
    n_cycles = int((valid != 0).sum())
    return n_cycles if n_cycles > 0 else None


def throughput_from_current(df: pd.DataFrame, current_col: str, time_col: str) -> Optional[float]:
    """Calculate energy throughput from current and time."""
    if current_col not in df.columns or time_col not in df.columns:
        return None
    try:
        current = pd.to_numeric(df[current_col], errors='coerce')
        time = pd.to_numeric(df[time_col], errors='coerce')
        # Integrate current over time (simplified)
        total = np.trapz(current, x=time)
        return float(abs(total))
    except Exception:
        return None


def arrange_chronologically(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Sort DataFrame by date column."""
    if date_col not in df.columns:
        return df
    dates = pd.to_datetime(df[date_col], errors='coerce')
    return df.loc[dates.argsort()].reset_index(drop=True)