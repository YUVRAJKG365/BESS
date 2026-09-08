import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from utils.helpers import (
    safe_float, safe_int, parse_date, format_date_display,
    temperature_stats, cycle_count_from_events, dod_from_charge_discharge
)
from utils.constants import EOL_SOH_THRESHOLD


class ValidationResult:
    """Result of validating a dataset."""
    
    def __init__(self):
        self.valid = True
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.file_ok = True
        self.schema_ok = False
        self.date_ok = False
        self.numeric_ok = False
        self.missing_ok = False
        self.duplicate_ok = False
        self.physical_ok = False
        self.model_compat_ok = False
    
    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False
    
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
    
    def is_required_missing(self) -> bool:
        """Check if any required validation failed."""
        return len(self.errors) > 0 or not self.schema_ok or not self.date_ok
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "valid": self.valid and len(self.errors) == 0,
            "file_ok": self.file_ok,
            "schema_ok": self.schema_ok,
            "date_ok": self.date_ok,
            "numeric_ok": self.numeric_ok,
            "missing_ok": self.missing_ok,
            "duplicate_ok": self.duplicate_ok,
            "physical_ok": self.physical_ok,
            "model_compat_ok": self.model_compat_ok,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class DataValidator:
    """Validates uploaded BESS datasets against required schemas."""
    
    def __init__(self, mapped_data: Dict[str, Any], oem_data: Optional[pd.DataFrame] = None):
        self.mapped_data = mapped_data  # Contains the mapped columns and cleaned data
        self.oem_data = oem_data
        self.result = ValidationResult()
    
    def validate_all(self) -> ValidationResult:
        """Run all validation checks."""
        self._validate_schema()
        self._validate_dates()
        self._validate_numeric()
        self._validate_missing()
        self._validate_duplicates()
        self._validate_physical()
        self._validate_model_compatibility()
        self.result.valid = len(self.result.errors) == 0 and self.result.schema_ok and self.result.date_ok
        return self.result
    
    def _validate_schema(self) -> None:
        """A. Schema validation - check required columns exist."""
        required_for_model = ['soh_pct', 'time', 'temperature']
        mapped_cols = self.mapped_data.get('mapped_columns') or {}
        clean_data = self.mapped_data.get('clean_data', pd.DataFrame())
        
        # If mapped_cols is missing any required variable, attempt auto-detection from clean_data
        if not all(r in mapped_cols and mapped_cols[r] for r in required_for_model):
            from backend.data_mapper import DataMapper
            auto = DataMapper().auto_detect_mappings(clean_data)
            for r, c in auto.items():
                if r not in mapped_cols or not mapped_cols[r]:
                    mapped_cols[r] = c
            self.mapped_data['mapped_columns'] = mapped_cols
        
        for req in required_for_model:
            col_mapped = mapped_cols.get(req)
            col_exists = (col_mapped and col_mapped in clean_data.columns) or (req in clean_data.columns)
            
            # Fallback checks for standard column names in clean_data
            if req == 'time' and not col_exists:
                for c in ['time_parsed', 'Date', 'date', 'Timestamp', 'timestamp', 'Time', 'time']:
                    if c in clean_data.columns:
                        col_exists = True
                        mapped_cols['time'] = c
                        break
            
            if req == 'soh_pct' and not col_exists:
                for c in ['soh_pct', 'SOH_pct', 'SOH', 'Battery SOH']:
                    if c in clean_data.columns:
                        col_exists = True
                        mapped_cols['soh_pct'] = c
                        break
                        
            if req == 'temperature' and not col_exists:
                for c in ['temperature', 'Avg_Temp_C', 'T_avg_C', 'Temperature']:
                    if c in clean_data.columns:
                        col_exists = True
                        mapped_cols['temperature'] = c
                        break
            
            if not col_exists:
                self.result.add_error(f"Missing required model variable: {req}")
        
        # SOH check specifically
        soh_present = (
            ('soh_pct' in mapped_cols and mapped_cols['soh_pct'] in clean_data.columns) or
            'soh_pct' in clean_data.columns or
            'SOH_pct' in clean_data.columns
        )
        if not soh_present:
            self.result.add_error("SOH column is required for degradation forecasting")
        
        self.result.schema_ok = len([e for e in self.result.errors if 'Missing required' in e or 'SOH column' in e]) == 0
    
    def _validate_dates(self) -> None:
        """C. Date/time validation."""
        df = self.mapped_data.get('clean_data', pd.DataFrame())
        mapped_cols = self.mapped_data.get('mapped_columns', {})
        time_col = mapped_cols.get('time')
        
        if not time_col or time_col not in df.columns:
            for fallback in ['time_parsed', 'Date', 'date', 'Timestamp', 'timestamp', 'Time', 'time']:
                if fallback in df.columns:
                    time_col = fallback
                    mapped_cols['time'] = fallback
                    break
        
        if time_col is None or time_col not in df.columns:
            self.result.add_error("Valid time column is required")
            self.result.date_ok = False
            return
        
        parsed = pd.to_datetime(df[time_col], errors='coerce')
        null_count = parsed.isna().sum()
        
        if len(df) > 0 and null_count > len(df) * 0.5:
            self.result.add_error("More than 50% of time values could not be parsed")
            self.result.date_ok = False
        else:
            self.result.date_ok = True
        
        # Check date range
        valid_dates = parsed.dropna()
        if len(valid_dates) < 3:
            self.result.add_error("Insufficient date observations (need at least 3)")
            self.result.date_ok = False
    
    def _validate_numeric(self) -> None:
        """D. Numeric validation."""
        df = self.mapped_data.get('clean_data', pd.DataFrame())
        mapped_cols = self.mapped_data.get('mapped_columns', {})
        
        numeric_errors = 0
        for var_name, col_key in mapped_cols.items():
            if var_name == 'time':
                continue  # Validated in _validate_dates
            if col_key and col_key in df.columns:
                series = pd.to_numeric(df[col_key], errors='coerce')
                null_after = series.isna().sum()
                if len(df) > 0 and null_after > len(df) * 0.5:
                    self.result.add_error(f"More than 50% of values in {var_name} ({col_key}) are non-numeric")
                    numeric_errors += 1
        
        self.result.numeric_ok = numeric_errors == 0
    
    def _validate_missing(self) -> None:
        """E. Missing-value validation."""
        df = self.mapped_data.get('clean_data', pd.DataFrame())
        mapped_cols = self.mapped_data.get('mapped_columns', {})
        
        critical_cols = []
        for var_name, col_key in mapped_cols.items():
            if col_key and col_key in df.columns:
                critical_cols.append(col_key)
        
        if not critical_cols:
            for cand in ['soh_pct', 'SOH_pct', 'temperature', 'Avg_Temp_C', 'SOC_pct', 'Avg_SOC_pct', 'Avg_DoD_pct']:
                if cand in df.columns:
                    critical_cols.append(cand)
        
        if not critical_cols:
            self.result.add_warning("No critical columns to validate missing values for")
            self.result.missing_ok = True
            return
        
        has_severe_missing = False
        for col in critical_cols:
            series = df[col]
            missing_pct = (series.isnull().sum() / len(series)) * 100 if len(series) > 0 else 100
            if missing_pct > 30:
                self.result.add_warning(f"Column {col} has {missing_pct:.1f}% missing values")
            if missing_pct > 80:
                has_severe_missing = True
        
        self.result.missing_ok = not has_severe_missing
    
    def _validate_duplicates(self) -> None:
        """F. Duplicate validation."""
        df = self.mapped_data.get('clean_data', pd.DataFrame())
        if len(df) > 0:
            dupes = df.duplicated().sum()
            dupe_pct = (dupes / len(df)) * 100
            if dupe_pct > 10:
                self.result.add_warning(f"Dataset has {dupe_pct:.1f}% duplicate rows")
            elif dupe_pct > 35:
                self.result.add_warning(f"Dataset has {dupe_pct:.1f}% duplicate rows - may affect calibration")
        
        self.result.duplicate_ok = True
    
    def _validate_physical(self) -> None:
        """G. Physical plausibility validation."""
        df = self.mapped_data.get('clean_data', pd.DataFrame())
        mapped_cols = self.mapped_data.get('mapped_columns', {})
        
        # SOH should be within reasonable bounds (0-100%)
        soh_col = mapped_cols.get('soh_pct')
        if not soh_col or soh_col not in df.columns:
            for c in ['soh_pct', 'SOH_pct', 'SOH']:
                if c in df.columns:
                    soh_col = c
                    break
        if soh_col and soh_col in df.columns:
            soh_values = pd.to_numeric(df[soh_col], errors='coerce').dropna()
            if len(soh_values) > 0:
                min_soh = soh_values.min()
                max_soh = soh_values.max()
                if min_soh < 0:
                    self.result.add_warning(f"SOH values go below 0% (min: {min_soh:.1f}%)")
                if max_soh > 100:
                    self.result.add_warning(f"SOH values go above 100% (max: {max_soh:.1f}%)")
        
        # Temperature plausibility
        temp_col = mapped_cols.get('temperature')
        if not temp_col or temp_col not in df.columns:
            for c in ['temperature', 'Avg_Temp_C', 'T_avg_C']:
                if c in df.columns:
                    temp_col = c
                    break
        if temp_col and temp_col in df.columns:
            temp_values = pd.to_numeric(df[temp_col], errors='coerce').dropna()
            if len(temp_values) > 0:
                min_temp = temp_values.min()
                max_temp = temp_values.max()
                if min_temp < -40:
                    self.result.add_warning(f"Minimum temperature very low: {min_temp:.1f}°C")
                if max_temp > 80:
                    self.result.add_warning(f"Maximum temperature very high: {max_temp:.1f}°C")
        
        # SOC plausibility
        soc_col = mapped_cols.get('SOC_pct')
        if not soc_col or soc_col not in df.columns:
            for c in ['SOC_pct', 'Avg_SOC_pct']:
                if c in df.columns:
                    soc_col = c
                    break
        if soc_col and soc_col in df.columns:
            soc_values = pd.to_numeric(df[soc_col], errors='coerce').dropna()
            if len(soc_values) > 0:
                min_soc = soc_values.min()
                max_soc = soc_values.max()
                if min_soc < 0 or max_soc > 100:
                    self.result.add_warning(f"SOC values outside 0-100% range ({min_soc:.1f}% to {max_soc:.1f}%)")
        
        self.result.physical_ok = True
    
    def _validate_model_compatibility(self) -> None:
        """H. Model compatibility validation."""
        mapped_cols = self.mapped_data.get('mapped_columns', {})
        df = self.mapped_data.get('clean_data', pd.DataFrame())
        
        # Check that we have enough temporal resolution
        time_col = mapped_cols.get('time')
        if not time_col or time_col not in df.columns:
            for c in ['time_parsed', 'Date', 'date', 'Timestamp', 'time']:
                if c in df.columns:
                    time_col = c
                    break
        if time_col and time_col in df.columns:
            parsed = pd.to_datetime(df[time_col], errors='coerce')
            valid = parsed.dropna()
            if len(valid) > 0:
                date_span = (valid.max() - valid.min()).days
                if date_span < 1:
                    self.result.add_warning("Dataset spans less than 1 day - calibration may be limited")
                elif date_span > 3650:  # ~10 years
                    self.result.add_warning("Dataset spans more than 10 years")
        
        # Check for at least some cycling data
        dod_col = mapped_cols.get('Avg_DoD_pct')
        if not dod_col or (dod_col not in df.columns and 'Avg_DoD_pct' not in df.columns):
            self.result.add_warning("No DoD variable mapped - cycle aging component may be limited")
        
        self.result.model_compat_ok = True