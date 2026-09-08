import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import os

from utils.helpers import (
    detect_file_type, safe_float, safe_int, parse_date,
    format_date_display, calculate_duration_hours, 
    soc_average, temperature_stats, cycle_count_from_events,
    throughput_from_current, arrange_chronologically
)
from utils.constants import EOL_SOH_THRESHOLD


class UploadedFile:
    """Represents a single uploaded file with its metadata."""
    
    def __init__(self, file_path: str, file_name: str, sheet_name: Optional[str] = None):
        self.file_path = file_path
        self.file_name = file_name
        self.sheet_name = sheet_name
        self.file_type: str = detect_file_type(file_name)
        self.dataframe: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {
            "rows": 0,
            "columns": 0,
            "date_range": (None, None),
            "variables": [],
            "missing_values": 0,
            "duplicate_rows": 0,
            "detected_soh_column": None,
            "detected_temperature_columns": [],
            "detected_soc_column": None,
            "detected_dod_column": None,
            "detected_cycles_column": None,
            "detected_efc_column": None,
            "detected_charge_duration": None,
            "detected_discharge_duration": None,
        }
        
    def load(self) -> bool:
        """Load the file into a DataFrame."""
        try:
            if self.file_type == 'excel':
                if self.sheet_name:
                    self.dataframe = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
                else:
                    # Try to auto-detect the best sheet, use first sheet
                    xls = pd.ExcelFile(self.file_path)
                    self.dataframe = pd.read_excel(self.file_path, sheet_name=xls.sheet_names[0])
                    self.sheet_name = xls.sheet_names[0]
            elif self.file_type == 'csv':
                self.dataframe = pd.read_csv(self.file_path)
            else:
                return False
            
            if self.dataframe is None or self.dataframe.empty:
                return False
                
            self._compute_metadata()
            return True
        except Exception as e:
            # File is corrupted or unsupported
            return False
    
    def _compute_metadata(self) -> None:
        """Compute metadata about the uploaded file."""
        if self.dataframe is None:
            return
            
        df = self.dataframe
        self.metadata["rows"] = len(df)
        self.metadata["columns"] = len(df.columns)
        self.metadata["variables"] = list(df.columns)
        
        # Date range
        date_cols = self._detect_date_columns()
        if date_cols:
            min_date, max_date = calculate_dates_range(df, date_cols[0])
            self.metadata["date_range"] = (min_date, max_date)
        
        # Missing values
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        self.metadata["missing_values"] = int(missing_cells) if total_cells > 0 else 0
        
        # Duplicate rows
        self.metadata["duplicate_rows"] = int(df.duplicated().sum())
        
        # Detect columns
        self._detect_column_types()
    
    def _detect_date_columns(self) -> List[str]:
        """Detect potential date columns."""
        date_cols = []
        for col in self.dataframe.columns:
            sample = pd.to_numeric(self.dataframe[col], errors='coerce')
            # If more than half the values parse as dates, it's a date column
            parsed_dates = pd.to_datetime(self.dataframe[col], errors='coerce')
            non_null_dates = parsed_dates.dropna()
            if len(non_null_dates) > 0 and len(non_null_dates) / len(self.dataframe) > 0.5:
                date_cols.append(col)
        return date_cols
    
    def _detect_column_types(self) -> None:
        """Detect SOH, temperature, SOC, DoD, cycles, EFC columns."""
        df = self.dataframe
        col_lower = {col: str(col).lower().strip() for col in df.columns}
        
        # SOH detection
        soh_cols = [col for col in df.columns if any(kw in col_lower[col] for kw in ['soh', 'health', 'state of health'])]
        if soh_cols:
            self.metadata["detected_soh_column"] = soh_cols[0]
        
        # Temperature columns (multiple possible)
        temp_cols = []
        for col in df.columns:
            cl = col_lower[col]
            if any(kw in cl for kw in ['temp', 'temperature', 't_', 'ct', 'ambient']):
                temp_cols.append(col)
        self.metadata["detected_temperature_columns"] = temp_cols[:3]  # limit to 3
        
        # SOC detection
        soc_cols = [col for col in df.columns if any(kw in col_lower[col] for kw in ['soc', 'state of charge'])]
        if soc_cols:
            self.metadata["detected_soc_column"] = soc_cols[0]
        
        # DoD detection
        dod_cols = [col for col in df.columns if any(kw in col_lower[col] for kw in ['dod', 'depth of discharge', 'dod%'])]
        if dod_cols:
            self.metadata["detected_dod_column"] = dod_cols[0]
        
        # Cycles per day
        cycles_cols = [col for col in df.columns if any(kw in col_lower[col] for kw in ['cycle', 'cycles', 'cycle count'])]
        if cycles_cols:
            self.metadata["detected_cycles_column"] = cycles_cols[0]
        
        # EFC per day
        efc_cols = [col for col in df.columns if any(kw in col_lower[col] for kw in ['efc', 'energy flow', 'throughput', 'daily'])]
        if efc_cols:
            self.metadata["detected_efc_column"] = efc_cols[0]
        
        # Charge duration
        charge_dur_cols = [col for col in df.columns if any(kw in col_lower[col] for kw in ['charge dur', 'charge_time', 'charge_h'])]
        if charge_dur_cols:
            self.metadata["detected_charge_duration"] = charge_dur_cols[0]
        
        # Discharge duration
        discharge_dur_cols = [col for col in df.columns if any(kw in col_lower[col] for kw in ['discharge dur', 'discharge_time', 'discharge_h'])]
        if discharge_dur_cols:
            self.metadata["detected_discharge_duration"] = discharge_dur_cols[0]