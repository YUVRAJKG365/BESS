import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Tuple


# Mapping dictionary: uploaded column name -> model variable name
COLUMN_MAPPINGS = {
    # SOH mappings
    "Battery SOH (%)": "soh_pct",
    "Battery SOH": "soh_pct",
    "SOH (%)": "soh_pct",
    "State of Health (%)": "soh_pct",
    "SoH": "soh_pct",
    "SOH_pct": "soh_pct",
    "SOH_fraction": "soh_pct",
    
    # Temperature mappings
    "Temperature (°C)": "temperature",
    "Temperature (C)": "temperature",
    "Temperature": "temperature",
    "Ambient Temp": "temperature",
    "Cell Temperature": "temperature",
    "Avg_Temp_C": "temperature",
    "Avg Temp C": "temperature",
    "AvgTemp_C": "temperature",
    
    # SOC mappings
    "SOC (%)": "SOC_pct",
    "State of Charge (%)": "SOC_pct",
    "SoC": "SOC_pct",
    "State of Charge": "SOC_pct",
    "Avg_SOC_pct": "SOC_pct",
    "Avg_SOC": "SOC_pct",
    "Avg_SoC_pct": "SOC_pct",
    
    # DoD mappings
    "Avg DoD (%)": "Avg_DoD_pct",
    "Depth of Discharge (%)": "Avg_DoD_pct",
    "DoD (%)": "Avg_DoD_pct",
    "Average DoD": "Avg_DoD_pct",
    "Avg_DoD_pct": "Avg_DoD_pct",
    
    # EFC/day mappings
    "Cycles/day": "Cycles_per_day",
    "Cycles per Day": "Cycles_per_day",
    "EFC/day": "EFC_per_day",
    "EFC per Day": "EFC_per_day",
    "Equivalent Full Cycles": "EFC_per_day",
    "EFC_per_day": "EFC_per_day",
    
    # Charge duration
    "Charge Duration (hr)": "Charge_Duration_hr",
    "Charge Duration": "Charge_Duration_hr",
    "Charge Time (hr)": "Charge_Duration_hr",
    "Charge_Duration_hr": "Charge_Duration_hr",
    
    # Discharge duration
    "Discharge Duration (hr)": "Discharge_Duration_hr",
    "Discharge Duration": "Discharge_Duration_hr",
    "Discharge Time (hr)": "Discharge_Duration_hr",
    "Discharge_Duration_hr": "Discharge_Duration_hr",
    
    # C-rate
    "C-rate": "C_rate",
    "C Rate": "C_rate",
    "Charge C-rate": "C_rate",
    "C_rate": "C_rate",
    "Rated_C_rate": "C_rate",
    "Rated C rate": "C_rate",
    
    # Max/Min temperature
    "Max_Temp_C": "Tmax",
    "Max Temp C": "Tmax",
    "MaxTemp_C": "Tmax",
    "Max Temperature": "Tmax",
    "Min_Temp_C": "Tmin",
    "Min Temp C": "Tmin",
    "MinTemp_C": "Tmin",
    "Min Temperature": "Tmin",
    
    # Time/stamp
    "Time": "time",
    "Date": "time",
    "Timestamp": "time",
    "Date/Time": "time",
    "Time (hr)": "time",
    
    # Additional variables
    "Power (kW)": "power_kw",
    "Voltage (V)": "voltage_v",
    "Current (A)": "current_a",
    
    # OEM columns
    "OEM_SOH_at_same_age_fraction": "oem_soh_fraction",
    "OEM_SOH_at_same_age_pct": "oem_soh_pct",
    "OEM_SOH": "oem_soh_pct",
}


class DataMapper:
    """Handles intelligent column mapping between uploaded files and model variables."""
    
    def __init__(self):
        self.mapped_columns: Dict[str, Optional[str]] = {}
        self.custom_mappings: Dict[str, str] = {}
        self.available_variables: List[str] = []
        
    def get_mapping_options(self, uploaded_columns: List[str]) -> List[Dict[str, str]]:
        """Get mapping options for uploaded columns."""
        options = []
        for col in uploaded_columns:
            # Check if there's a default mapping
            default_map = COLUMN_MAPPINGS.get(col)
            options.append({
                "uploaded_column": col,
                "model_variable": default_map if default_map else f"unset_{col}",
                "status": "default" if default_map else "unmapped",
            })
        return options
    
    def auto_detect_mappings(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect mappings between dataframe columns and model variables.
        Returns dict of {model_variable: dataframe_column_name}
        """
        mappings = {}
        cols = list(df.columns)
        
        def norm(s):
            return "".join(c for c in str(s).lower() if c.isalnum())
        
        col_norm_map = {norm(c): c for c in cols}
        
        var_candidates = {
            "time": [
                "date", "time", "timestamp", "datetime", "timeparsed", "timehr", "datehour"
            ],
            "soh_pct": [
                "sohpct", "soh", "batterysohpct", "batterysoh", "stateofhealthpct", "stateofhealth", "sohfraction"
            ],
            "temperature": [
                "avgtempc", "temperaturec", "temperature", "ambienttemp", "celltemperature", "avgtemp", "tavgc", "tavg"
            ],
            "SOC_pct": [
                "avgsocpct", "socpct", "soc", "stateofchargepct", "stateofcharge", "avgsoc"
            ],
            "Avg_DoD_pct": [
                "avgdodpct", "dodpct", "dod", "depthofdischargepct", "depthofdischarge", "avgdod"
            ],
            "EFC_per_day": [
                "efcperday", "efcday", "efc", "equivalentfullcycles", "dailyefc"
            ],
            "Cycles_per_day": [
                "cyclesperday", "cyclesday", "cycles", "dailycycles"
            ],
            "C_rate": [
                "ratedcrate", "crate", "chargecrate", "dischargecrate"
            ],
            "Tmax": [
                "maxtempc", "maxtemp", "maxtemperature", "tmax", "tmaxc"
            ],
            "Tmin": [
                "mintempc", "mintemp", "mintemperature", "tmin", "tminc"
            ],
            "Charge_Duration_hr": [
                "chargedurationhr", "chargeduration", "chargetimehr", "chargetime"
            ],
            "Discharge_Duration_hr": [
                "dischargedurationhr", "dischargeduration", "dischargetimehr", "dischargetime"
            ],
        }
        
        for model_var, candidates in var_candidates.items():
            for cand in candidates:
                if cand in col_norm_map:
                    mappings[model_var] = col_norm_map[cand]
                    break
                    
        self.mapped_columns = mappings.copy()
        return mappings
    
    def apply_mapping(self, mapping_choices: Dict[str, str]) -> Dict[str, Optional[str]]:
        """
        Apply user-selected mappings and return mapped columns dict.
        Supports both {uploaded_col: model_var} and {model_var: uploaded_col}.
        Always returns normalized dict of {model_var: uploaded_col}.
        """
        model_vars = {
            "soh_pct", "temperature", "SOC_pct", "Avg_DoD_pct",
            "EFC_per_day", "Cycles_per_day", "Charge_Duration_hr",
            "Discharge_Duration_hr", "C_rate", "time", "Tmax", "Tmin",
            "power_kw", "voltage_v", "current_a", "oem_soh_pct", "oem_soh_fraction"
        }
        self.mapped_columns = {}
        for k, v in (mapping_choices or {}).items():
            if k in model_vars:
                self.mapped_columns[k] = v
            elif v in model_vars:
                self.mapped_columns[v] = k
            else:
                self.mapped_columns[str(v)] = k
        return self.mapped_columns
    
    def get_mapped_column(self, model_var: str) -> Optional[str]:
        """Get the uploaded column name mapped to a model variable."""
        return self.mapped_columns.get(model_var)
    
    def apply_custom_mapping(self, uploaded_col: str, model_var: str) -> None:
        """Apply a custom mapping (user-defined)."""
        self.mapped_columns[model_var] = uploaded_col
        self.custom_mappings[uploaded_col] = model_var
    
    def clean_data(self, df: pd.DataFrame, mapped_columns: Dict[str, Optional[str]]) -> pd.DataFrame:
        """Clean and format data based on mapped columns."""
        df_clean = df.copy()
        mapped_cols = mapped_columns or {}
        
        # Helper to find column by aliases
        def find_col(var_name: str, aliases: List[str]) -> Optional[str]:
            if var_name in mapped_cols and mapped_cols[var_name] in df_clean.columns:
                return mapped_cols[var_name]
            for a in aliases:
                if a in df_clean.columns:
                    return a
            lower_map = {str(c).lower(): c for c in df_clean.columns}
            for a in aliases:
                if a.lower() in lower_map:
                    return lower_map[a.lower()]
            return None

        # 1. Time / Date
        time_c = find_col('time', ['Date', 'date', 'DATE', 'Timestamp', 'timestamp', 'Time', 'time'])
        if time_c:
            df_clean['time_parsed'] = pd.to_datetime(df_clean[time_c], errors='coerce')
        elif 'time_parsed' not in df_clean.columns:
            df_clean['time_parsed'] = pd.date_range(start='2022-01-01', periods=len(df_clean), freq='D')
            
        t0 = df_clean['time_parsed'].min()
        df_clean['time_hours'] = (df_clean['time_parsed'] - t0).dt.total_seconds() / 3600.0

        # 2. SOH
        soh_c = find_col('soh_pct', ['SOH_pct', 'SOH_Pct', 'soh_pct', 'SOH', 'Battery SOH (%)', 'Battery SOH', 'State of Health (%)', 'SoH'])
        if soh_c:
            df_clean['soh_pct'] = pd.to_numeric(df_clean[soh_c], errors='coerce')
        elif 'SOH_fraction' in df_clean.columns:
            df_clean['soh_pct'] = pd.to_numeric(df_clean['SOH_fraction'], errors='coerce') * 100.0
        else:
            df_clean['soh_pct'] = 100.0
        df_clean['SOH_pct'] = df_clean['soh_pct']

        # 3. Temperature
        temp_c = find_col('temperature', ['Avg_Temp_C', 'avg_temp_c', 'T_avg_C', 'Temperature (°C)', 'Temperature (C)', 'Temperature', 'AvgTemp_C'])
        if temp_c:
            df_clean['temperature'] = pd.to_numeric(df_clean[temp_c], errors='coerce')
        else:
            df_clean['temperature'] = 25.0
        df_clean['Avg_Temp_C'] = df_clean['temperature']
        df_clean['T_avg_C'] = df_clean['temperature']

        # 4. Tmax & Tmin
        tmax_c = find_col('Tmax', ['Max_Temp_C', 'max_temp_c', 'T_max_C', 'Max Temperature', 'MaxTemp_C'])
        if tmax_c:
            df_clean['Tmax'] = pd.to_numeric(df_clean[tmax_c], errors='coerce')
        else:
            df_clean['Tmax'] = df_clean['temperature']
        df_clean['Max_Temp_C'] = df_clean['Tmax']
        df_clean['T_max_C'] = df_clean['Tmax']

        tmin_c = find_col('Tmin', ['Min_Temp_C', 'min_temp_c', 'T_min_C', 'Min Temperature', 'MinTemp_C'])
        if tmin_c:
            df_clean['Tmin'] = pd.to_numeric(df_clean[tmin_c], errors='coerce')
        else:
            df_clean['Tmin'] = df_clean['temperature']
        df_clean['Min_Temp_C'] = df_clean['Tmin']
        df_clean['T_min_C'] = df_clean['Tmin']

        # 5. SOC
        soc_c = find_col('SOC_pct', ['Avg_SOC_pct', 'Avg_SOC', 'SOC_avg_pct', 'SOC_pct', 'avg_soc_pct', 'State of Charge (%)', 'SOC (%)', 'Avg_SoC_pct'])
        if soc_c:
            df_clean['SOC_pct'] = pd.to_numeric(df_clean[soc_c], errors='coerce')
        else:
            df_clean['SOC_pct'] = 50.0
        df_clean['Avg_SOC_pct'] = df_clean['SOC_pct']

        # 6. DoD
        dod_c = find_col('Avg_DoD_pct', ['Avg_DoD_pct', 'Avg_DOD_pct', 'Avg_DoD', 'DOD_pct', 'DoD_pct', 'avg_dod_pct', 'Depth of Discharge (%)', 'Avg DoD (%)'])
        if dod_c:
            df_clean['Avg_DoD_pct'] = pd.to_numeric(df_clean[dod_c], errors='coerce').clip(lower=1.0, upper=100.0)
        else:
            df_clean['Avg_DoD_pct'] = 95.0
        df_clean['DOD_actual'] = df_clean['Avg_DoD_pct'] / 100.0

        # 7. Charge & Discharge Duration
        chg_c = find_col('Charge_Duration_hr', ['Charge_Duration_hr', 'Charge_Duration', 'Charge_hr', 'Charge Duration (hr)'])
        df_clean['Charge_Duration_hr'] = pd.to_numeric(df_clean[chg_c], errors='coerce').fillna(4.0) if chg_c else 4.0
        
        dis_c = find_col('Discharge_Duration_hr', ['Discharge_Duration_hr', 'Discharge_Duration', 'Discharge_hr', 'Discharge Duration (hr)'])
        df_clean['Discharge_Duration_hr'] = pd.to_numeric(df_clean[dis_c], errors='coerce').fillna(4.0) if dis_c else 4.0

        # 8. EFC & Cycles per day
        efc_c = find_col('EFC_per_day', ['EFC_per_day', 'efc_per_day', 'EFC/day', 'EFC per Day'])
        cyc_c = find_col('Cycles_per_day', ['Cycles_per_day', 'cycles_per_day', 'Cycles/day', 'Cycles', 'Cycles per Day'])
        
        if efc_c:
            df_clean['EFC_per_day'] = pd.to_numeric(df_clean[efc_c], errors='coerce')
        if cyc_c:
            df_clean['Cycles_per_day'] = pd.to_numeric(df_clean[cyc_c], errors='coerce')

        if 'EFC_per_day' in df_clean.columns and 'Cycles_per_day' not in df_clean.columns:
            df_clean['Cycles_per_day'] = df_clean['EFC_per_day'] / df_clean['DOD_actual'].replace(0, np.nan).fillna(0.95)
        elif 'Cycles_per_day' in df_clean.columns and 'EFC_per_day' not in df_clean.columns:
            df_clean['EFC_per_day'] = df_clean['Cycles_per_day'] * df_clean['DOD_actual']
        elif 'EFC_per_day' not in df_clean.columns and 'Cycles_per_day' not in df_clean.columns:
            df_clean['EFC_per_day'] = 1.0
            df_clean['Cycles_per_day'] = 1.0

        # 9. C-rate
        cr_c = find_col('C_rate', ['Rated_C_rate', 'Rated C rate', 'C_rate', 'C-rate', 'Charge C-rate'])
        if cr_c:
            df_clean['C_rate'] = pd.to_numeric(df_clean[cr_c], errors='coerce')
        else:
            df_clean['C_rate'] = df_clean['DOD_actual'] / df_clean['Charge_Duration_hr'].replace(0, np.nan).fillna(4.0)

        return df_clean