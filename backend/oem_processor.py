import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from utils.helpers import parse_date, format_date_display, temperature_stats


@dataclass
class OEMData:
    """Container for OEM baseline data."""
    raw_data: pd.DataFrame
    time_column: str
    soh_column: str
    processed_time: np.ndarray
    processed_soh: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._compute_metadata()
    
    def _compute_metadata(self) -> None:
        """Compute OEM baseline metadata."""
        if len(self.processed_time) > 0 and len(self.processed_soh) > 0:
            self.metadata = {
                "time_span_years": (self.processed_time[-1] - self.processed_time[0]) / (365.25 * 24) if len(self.processed_time) > 1 else 0,
                "initial_soh": float(self.processed_soh[0]) if len(self.processed_soh) > 0 else 100.0,
                "final_soh": float(self.processed_soh[-1]) if len(self.processed_soh) > 0 else 100.0,
                "total_degradation": float(self.processed_soh[0] - self.processed_soh[-1]) if len(self.processed_soh) > 1 else 0,
                "n_points": len(self.processed_time),
                "degradation_rate_pct_per_year": 0.0,
            }
            
            if self.metadata["time_span_years"] > 0:
                self.metadata["degradation_rate_pct_per_year"] = (
                    self.metadata["total_degradation"] / self.metadata["time_span_years"]
                )
                
    @property
    def soh_pct(self) -> np.ndarray:
        return self.processed_soh
        
    @property
    def time_hours(self) -> np.ndarray:
        return self.processed_time
        
    def to_dict(self) -> Dict[str, Any]:
        return self.metadata
        
    def get_visualization_data(self) -> Dict[str, Any]:
        time_col = self.time_column
        raw_time = self.raw_data[time_col] if time_col in self.raw_data.columns else pd.Series()
        time_parsed = pd.to_datetime(raw_time, errors='coerce')
        if time_parsed.notna().sum() > len(raw_time) * 0.5:
            dates = time_parsed.values
        else:
            base = pd.Timestamp("2024-01-01")
            dates = (base + pd.to_timedelta(self.processed_time, unit='h')).values
            
        return {
            "dates": dates,
            "days": self.processed_time / 24.0,
            "years": self.processed_time / (365.25 * 24.0),
            "soh_pct": self.processed_soh,
        }


class OEMProcessor:
    """Processes and validates OEM baseline data."""
    
    def __init__(self):
        self.oem_data: Optional[OEMData] = None
        self.column_mapping: Dict[str, str] = {}
    
    def load_oem_file(self, 
                      file_path: Any, 
                      sheet_name: Optional[str] = None,
                      time_col: Optional[str] = None,
                      soh_col: Optional[str] = None) -> bool:
        """Load OEM baseline from file path, buffer, or DataFrame."""
        try:
            if isinstance(file_path, pd.DataFrame):
                df = file_path.copy()
            else:
                name = getattr(file_path, 'name', str(file_path))
                ext = name.lower().split('.')[-1] if '.' in name else ''
                
                if ext in ['xlsx', 'xls'] or hasattr(file_path, 'read'):
                    try:
                        xls = pd.ExcelFile(file_path)
                        if sheet_name and sheet_name in xls.sheet_names:
                            target_sheet = sheet_name
                        elif "OEM_Baseline" in xls.sheet_names:
                            target_sheet = "OEM_Baseline"
                        elif "Sheet1" in xls.sheet_names:
                            target_sheet = "Sheet1"
                        else:
                            target_sheet = xls.sheet_names[0]
                        df = pd.read_excel(file_path, sheet_name=target_sheet)
                    except Exception:
                        if hasattr(file_path, 'seek'):
                            file_path.seek(0)
                        df = pd.read_csv(file_path)
                elif ext == 'csv':
                    df = pd.read_csv(file_path)
                else:
                    return False
            
            # Auto-detect columns if not specified
            if time_col is None:
                time_col = self._detect_time_column(df)
            if soh_col is None:
                soh_col = self._detect_soh_column(df)
            
            if time_col is None or soh_col is None:
                return False
            
            # Process data
            self.oem_data = self._process_oem_data(df, time_col, soh_col)
            self.column_mapping = {"time": time_col, "soh": soh_col}
            
            return True
        except Exception:
            return False
    
    def _detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect time column."""
        candidates = ['day', 'date', 'timestamp', 'time', 'year', 'month', 'cycle', 'cycles']
        for c_pattern in candidates:
            for col in df.columns:
                cl = str(col).lower()
                if c_pattern == cl or (c_pattern in cl and not any(skip in cl for skip in ['soh', 'temp', 'soc', 'dod', 'rate'])):
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() > len(df) * 0.5:
                        return col
                    numeric = pd.to_numeric(df[col], errors='coerce')
                    if numeric.notna().sum() > len(df) * 0.5:
                        return col
        return None
    
    def _detect_soh_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect SOH column."""
        candidates = ['oem_soh_pct', 'soh_pct', 'oem_soh', 'soh', 'health', 'state of health', 'capacity', 'retention']
        for c_pattern in candidates:
            for col in df.columns:
                cl = str(col).lower()
                if c_pattern in cl and 'fraction' not in cl:
                    numeric = pd.to_numeric(df[col], errors='coerce')
                    if numeric.notna().sum() > len(df) * 0.5:
                        return col
        return None
    
    def _process_oem_data(self, df: pd.DataFrame, time_col: str, soh_col: str) -> OEMData:
        """Process OEM data into standard format."""
        df = df.copy()
        
        # Parse SOH
        soh_parsed = pd.to_numeric(df[soh_col], errors='coerce')
        
        # Parse time: check if numeric or calendar date
        is_num = pd.api.types.is_numeric_dtype(df[time_col]) or df[time_col].astype(str).str.replace('.', '', regex=False).str.isdigit().all()
        cl_name = time_col.lower()
        
        if is_num or cl_name in ['day', 'days', 'year', 'years', 'month', 'months', 'hour', 'hours']:
            time_num = pd.to_numeric(df[time_col], errors='coerce')
            if 'year' in cl_name:
                time_hours = time_num * 365.25 * 24.0
            elif 'day' in cl_name:
                time_hours = time_num * 24.0
            elif 'month' in cl_name:
                time_hours = time_num * 30.4375 * 24.0
            elif 'hour' in cl_name:
                time_hours = time_num
            else:
                max_v = time_num.max() if len(time_num) > 0 else 0
                if max_v <= 30:
                    time_hours = time_num * 365.25 * 24.0
                elif max_v <= 15000:
                    time_hours = time_num * 24.0
                else:
                    time_hours = time_num
            valid = time_hours.notna() & soh_parsed.notna()
            time_clean = time_hours[valid].values
            soh_clean = soh_parsed[valid].values
        else:
            time_parsed = pd.to_datetime(df[time_col], errors='coerce')
            valid = time_parsed.notna() & soh_parsed.notna()
            t_dt = time_parsed[valid].values
            soh_clean = soh_parsed[valid].values
            if len(t_dt) > 0:
                t0 = t_dt[0]
                time_clean = (t_dt - t0).astype('timedelta64[h]').astype(float)
            else:
                time_clean = np.array([])
        
        # Sort by time
        if len(time_clean) > 1:
            sort_idx = np.argsort(time_clean)
            time_clean = time_clean[sort_idx]
            soh_clean = soh_clean[sort_idx]
        
        return OEMData(
            raw_data=df,
            time_column=time_col,
            soh_column=soh_col,
            processed_time=time_clean,
            processed_soh=soh_clean,
        )
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data formatted for visualization."""
        if self.oem_data is None:
            return {}
        
        time_col = self.column_mapping.get("time", "")
        raw_time = self.oem_data.raw_data[time_col] if time_col in self.oem_data.raw_data.columns else pd.Series()
        time_parsed = pd.to_datetime(raw_time, errors='coerce')
        if time_parsed.notna().sum() > len(raw_time) * 0.5:
            dates = time_parsed.values
        else:
            base = pd.Timestamp("2024-01-01")
            dates = (base + pd.to_timedelta(self.oem_data.processed_time, unit='h')).values
        
        return {
            "dates": dates,
            "days": self.oem_data.processed_time / 24.0,
            "years": self.oem_data.processed_time / (365.25 * 24.0),
            "soh_pct": self.oem_data.processed_soh,
            "time_hours": self.oem_data.processed_time,
            "metadata": self.oem_data.metadata,
        }
    
    def get_model_input(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get OEM data formatted for model calibration."""
        if self.oem_data is None:
            return np.array([]), np.array([])
        return self.oem_data.processed_time, self.oem_data.processed_soh
    
    def validate(self) -> Dict[str, Any]:
        """Validate OEM data quality."""
        if self.oem_data is None:
            return {"valid": False, "error": "No OEM data loaded"}
        
        result = {"valid": True, "warnings": [], "errors": []}
        
        n_points = len(self.oem_data.processed_time)
        if n_points < 3:
            result["errors"].append("OEM baseline has fewer than 3 data points")
            result["valid"] = False
        elif n_points < 10:
            result["warnings"].append(f"OEM baseline has only {n_points} points - calibration may be limited")
        
        # Check SOH monotonic decrease (mostly)
        soh = self.oem_data.processed_soh
        if len(soh) > 2:
            increases = np.diff(soh) > 2.0  # Increases > 2%
            if np.any(increases):
                result["warnings"].append(f"OEM SOH increases at {np.sum(increases)} points")
        
        # Check time span
        if self.oem_data.metadata.get("time_span_years", 0) < 0.5:
            result["warnings"].append("OEM baseline spans less than 6 months")
        
        return result


def load_oem_baseline(file_path: str, 
                      sheet_name: Optional[str] = None,
                      time_col: Optional[str] = None,
                      soh_col: Optional[str] = None) -> Optional[OEMData]:
    """Convenience function to load OEM baseline."""
    processor = OEMProcessor()
    success = processor.load_oem_file(file_path, sheet_name, time_col, soh_col)
    if success:
        return processor.oem_data
    return None