import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from backend.physics_model import (
    PhysicsDegradationModel, PhysicsParameters, 
    detect_soh_steps, smooth_soh_target, calculate_metrics
)
from utils.constants import EOL_SOH_THRESHOLD
from scipy import stats


@dataclass
class DiagnosticsResult:
    """Container for diagnostic information."""
    # Step events
    step_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Calibration diagnostics
    calibration_history: List[Dict] = field(default_factory=list)
    parameter_bounds_status: Dict[str, str] = field(default_factory=dict)
    parameter_values: Dict[str, float] = field(default_factory=dict)
    
    # Residuals
    residuals_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals_smooth: np.ndarray = field(default_factory=lambda: np.array([]))
    residual_statistics: Dict[str, float] = field(default_factory=dict)
    
    # Stress distributions
    stress_distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Data quality
    data_quality_warnings: List[str] = field(default_factory=list)
    model_warnings: List[str] = field(default_factory=list)
    
    # Scenario diagnostics
    scenario_diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # ML residual diagnostic (optional)
    ml_residual_diagnostic: Optional[Dict] = None


class DiagnosticsAnalyzer:
    """Generates technical diagnostics for the degradation model."""
    
    def __init__(self,
                 model: PhysicsDegradationModel,
                 field_data: pd.DataFrame,
                 oem_data: Optional[pd.DataFrame] = None):
        self.model = model
        self.field_data = field_data.copy()
        self.oem_data = oem_data
        self.result = DiagnosticsResult()
    
    def run_all_diagnostics(self) -> DiagnosticsResult:
        """Run all diagnostic analyses."""
        self._analyze_step_events()
        self._analyze_calibration()
        self._analyze_residuals()
        self._analyze_stress_distributions()
        self._analyze_data_quality()
        self._analyze_model_warnings()
        return self.result
    
    def _analyze_step_events(self) -> None:
        """Analyze SOH step events (sudden capacity drops)."""
        df = self.field_data.sort_values('time_parsed').reset_index(drop=True)
        soh = pd.to_numeric(df['soh_pct'], errors='coerce').values
        time_hours = pd.to_numeric(df['time_hours'], errors='coerce').values if 'time_hours' in df.columns else np.arange(len(soh))
        
        steps = detect_soh_steps(soh, time_hours, threshold_pct=1.0)
        
        for idx in steps:
            if idx > 0 and idx < len(soh) - 1:
                before = np.nanmean(soh[max(0, idx-5):idx])
                after = np.nanmean(soh[idx+1:idx+6])
                drop = before - after
                
                self.result.step_events.append({
                    "index": int(idx),
                    "date": str(df.iloc[idx]['time_parsed']) if 'time_parsed' in df.columns else "N/A",
                    "soh_before": float(before),
                    "soh_after": float(after),
                    "drop_pct": float(drop),
                    "time_hours": float(time_hours[idx]),
                    "classification": "artifact" if drop > 5 else "physical",
                })
    
    def _analyze_calibration(self) -> None:
        """Analyze calibration results."""
        if hasattr(self.model, 'calibration_history'):
            self.result.calibration_history = self.model.calibration_history
        
        from utils.constants import PARAM_BOUNDS
        param_dict = self.model.params.to_dict()
        self.result.parameter_values = param_dict
        
        for name, value in param_dict.items():
            if name in PARAM_BOUNDS:
                low, high = PARAM_BOUNDS[name]
                if abs(value - low) < 1e-6:
                    self.result.parameter_bounds_status[name] = "AT_LOWER_BOUND"
                elif abs(value - high) < 1e-6:
                    self.result.parameter_bounds_status[name] = "AT_UPPER_BOUND"
                else:
                    self.result.parameter_bounds_status[name] = "OK"
            else:
                self.result.parameter_bounds_status[name] = "UNBOUNDED"
    
    def _analyze_residuals(self) -> None:
        """Analyze model residuals."""
        df = self.field_data.sort_values('time_parsed').reset_index(drop=True)
        
        t0 = df['time_parsed'].min()
        time_hours = (df['time_parsed'] - t0).dt.total_seconds() / 3600.0
        soh_obs = pd.to_numeric(df['soh_pct'], errors='coerce').values
        
        soh_smooth = smooth_soh_target(soh_obs, time_hours.values)
        
        soh_pred, _, _ = self.model.predict_soh(
            time_hours=time_hours.values,
            temperature=pd.to_numeric(df['temperature'], errors='coerce').values,
            soc=pd.to_numeric(df['SOC_pct'], errors='coerce').values,
            efc=pd.to_numeric(df['EFC_per_day'], errors='coerce').values,
            dod=pd.to_numeric(df['Avg_DoD_pct'], errors='coerce').values,
            c_rate=pd.to_numeric(df['C_rate'], errors='coerce').values,
            tmax=pd.to_numeric(df['Tmax'], errors='coerce').values,
            tmin=pd.to_numeric(df['Tmin'], errors='coerce').values,
            soh0=100.0
        )
        
        valid = ~np.isnan(soh_obs) & ~np.isnan(soh_pred)
        self.result.residuals_raw = soh_obs[valid] - soh_pred[valid]
        
        valid_smooth = ~np.isnan(soh_smooth) & ~np.isnan(soh_pred)
        self.result.residuals_smooth = soh_smooth[valid_smooth] - soh_pred[valid_smooth]
        
        res = self.result.residuals_raw
        if len(res) > 0:
            self.result.residual_statistics = {
                "mean": float(np.mean(res)),
                "std": float(np.std(res)),
                "skewness": float(stats.skew(res)) if len(res) > 2 else 0,
                "kurtosis": float(stats.kurtosis(res)) if len(res) > 3 else 0,
                "autocorr_1": float(np.corrcoef(res[:-1], res[1:])[0, 1]) if len(res) > 1 else 0,
            }
        
        self.result.residual_statistics.update(calculate_metrics(soh_obs, soh_pred))
        self.result.residual_statistics.update(
            {f"vs_smooth_{k}": v for k, v in calculate_metrics(soh_smooth, soh_pred).items()}
        )
    
    def _analyze_stress_distributions(self) -> None:
        """Analyze distributions of stress factors."""
        df = self.field_data
        
        stress_vars = {
            'temperature': 'temperature',
            'SOC_pct': 'SOC_pct',
            'Avg_DoD_pct': 'Avg_DoD_pct',
            'EFC_per_day': 'EFC_per_day',
            'Cycles_per_day': 'Cycles_per_day',
            'C_rate': 'C_rate',
            'Tmax': 'Tmax',
            'Tmin': 'Tmin',
        }
        
        for name, col in stress_vars.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(vals) > 0:
                    self.result.stress_distributions[name] = {
                        "count": len(vals),
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "median": float(vals.median()),
                        "q25": float(vals.quantile(0.25)),
                        "q75": float(vals.quantile(0.75)),
                    }
    
    def _analyze_data_quality(self) -> None:
        """Analyze data quality issues."""
        df = self.field_data
        
        if 'time_parsed' in df.columns:
            df_sorted = df.sort_values('time_parsed')
            time_diff = df_sorted['time_parsed'].diff().dt.total_seconds() / 3600.0
            gaps = time_diff[time_diff > 48]
            if len(gaps) > 0:
                self.result.data_quality_warnings.append(
                    f"Found {len(gaps)} time gaps > 48 hours (max: {gaps.max():.1f} hours)"
                )
        
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 20:
                self.result.data_quality_warnings.append(
                    f"Column '{col}' has {missing_pct:.1f}% missing values"
                )
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() == 1:
                self.result.data_quality_warnings.append(
                    f"Column '{col}' has constant value ({df[col].iloc[0]})"
                )
        
        if 'soh_pct' in df.columns:
            soh = pd.to_numeric(df.sort_values('time_parsed')['soh_pct'], errors='coerce').dropna()
            if len(soh) > 2:
                increases = np.diff(soh) > 1.0
                if np.any(increases):
                    self.result.data_quality_warnings.append(
                        f"SOH increases detected at {np.sum(increases)} points"
                    )
    
    def _analyze_model_warnings(self) -> None:
        """Generate model-specific warnings."""
        for name, status in self.result.parameter_bounds_status.items():
            if status == "AT_LOWER_BOUND":
                self.result.model_warnings.append(f"Parameter {name} at lower bound - may be under-constrained")
            elif status == "AT_UPPER_BOUND":
                self.result.model_warnings.append(f"Parameter {name} at upper bound - may be over-constrained")
        
        hist_data = self._prepare_historical_data()
        soh_pred, cal_loss, cyc_loss = self.model.predict_soh(
            time_hours=hist_data['time_hours'],
            temperature=hist_data['temperature'],
            soc=hist_data['SOC_pct'],
            efc=hist_data['EFC_per_day'],
            dod=hist_data['Avg_DoD_pct'],
            c_rate=hist_data['C_rate'],
            tmax=hist_data['Tmax'],
            tmin=hist_data['Tmin'],
            soh0=100.0
        )
        
        if cal_loss[-1] > cyc_loss[-1] * 5:
            self.result.model_warnings.append("Calendar aging dominates - check temperature/SOC stress factors")
        elif cyc_loss[-1] > cal_loss[-1] * 5:
            self.result.model_warnings.append("Cycle aging dominates - check DoD/C-rate/EFC stress factors")
        
        from backend.physics_model import predict_eol
        n_days = 3650
        future_time = np.arange(hist_data['time_hours'][-1] + 24, 
                                hist_data['time_hours'][-1] + 24 + n_days * 24, 24.0)
        temp = np.full(n_days, np.nanmedian(hist_data['temperature']))
        soc = np.full(n_days, np.nanmedian(hist_data['SOC_pct']))
        dod = np.full(n_days, np.nanmedian(hist_data['Avg_DoD_pct']))
        efc = np.full(n_days, np.nanmedian(hist_data['EFC_per_day']))
        crate = np.full(n_days, np.nanmedian(hist_data['C_rate']))
        tmax = np.full(n_days, np.nanmedian(hist_data['Tmax']))
        tmin = np.full(n_days, np.nanmedian(hist_data['Tmin']))
        
        soh_future, _, _ = self.model.predict_soh(
            time_hours=future_time,
            temperature=temp, soc=soc, efc=efc, dod=dod,
            c_rate=crate, tmax=tmax, tmin=tmin, soh0=soh_pred[-1]
        )
        
        eol = predict_eol(soh_future, future_time, EOL_SOH_THRESHOLD)
        if eol['eol_reached'] and eol['years_to_eol'] < 2:
            self.result.model_warnings.append(f"EOL predicted within {eol['years_to_eol']:.1f} years")
    
    def _prepare_historical_data(self) -> Dict[str, np.ndarray]:
        """Prepare historical data arrays."""
        df = self.field_data.sort_values('time_parsed').reset_index(drop=True)
        t0 = df['time_parsed'].min()
        time_hours = (df['time_parsed'] - t0).dt.total_seconds() / 3600.0
        
        return {
            'time_hours': time_hours.values,
            'temperature': pd.to_numeric(df['temperature'], errors='coerce').values,
            'SOC_pct': pd.to_numeric(df['SOC_pct'], errors='coerce').values,
            'Avg_DoD_pct': pd.to_numeric(df['Avg_DoD_pct'], errors='coerce').values,
            'EFC_per_day': pd.to_numeric(df['EFC_per_day'], errors='coerce').values,
            'Cycles_per_day': pd.to_numeric(df['Cycles_per_day'], errors='coerce').values,
            'C_rate': pd.to_numeric(df['C_rate'], errors='coerce').values,
            'Tmax': pd.to_numeric(df['Tmax'], errors='coerce').values,
            'Tmin': pd.to_numeric(df['Tmin'], errors='coerce').values,
        }


def run_diagnostics(model: PhysicsDegradationModel,
                    field_data: pd.DataFrame,
                    oem_data: Optional[pd.DataFrame] = None) -> DiagnosticsResult:
    """Convenience function to run diagnostics."""
    analyzer = DiagnosticsAnalyzer(model, field_data, oem_data)
    return analyzer.run_all_diagnostics()