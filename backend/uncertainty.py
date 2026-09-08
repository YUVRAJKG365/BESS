import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from scipy import stats
from backend.physics_model import PhysicsDegradationModel, PhysicsParameters, predict_eol
from utils.constants import EOL_SOH_THRESHOLD


@dataclass
class UncertaintyResult:
    """Container for uncertainty analysis results."""
    eol_samples: np.ndarray
    eol_percentiles: Dict[str, float]
    soh_bands: Dict[str, np.ndarray]  # percentile -> soh array
    parameter_samples: List[PhysicsParameters]
    forecast_dates: np.ndarray
    method: str = "parameter_ensemble"
    warning: str = "Parameter ensemble shown here is a local sensitivity/uncertainty analysis and is not a formal statistical confidence interval."


class UncertaintyAnalyzer:
    """
    Performs parameter ensemble / uncertainty analysis following V11 methodology.
    
    Generates parameter ensembles by sampling around calibrated parameters
    and propagates through the model to get SOH uncertainty bands and EOL distributions.
    """
    
    def __init__(self,
                 model: PhysicsDegradationModel,
                 field_data: pd.DataFrame,
                 oem_data: Optional[pd.DataFrame] = None,
                 n_samples: int = 100,
                 parameter_perturbation: float = 0.2):
        """
        Initialize uncertainty analyzer.
        
        Args:
            model: Calibrated physics model
            field_data: Historical field data
            oem_data: Optional OEM baseline
            n_samples: Number of ensemble samples
            parameter_perturbation: Relative perturbation for parameter sampling (e.g., 0.2 = ±20%)
        """
        self.model = model
        self.field_data = field_data.copy()
        self.oem_data = oem_data
        self.n_samples = n_samples
        self.parameter_perturbation = parameter_perturbation
        self.calibrated_params = model.params
    
    def run_uncertainty_analysis(self, forecast_years: float = 10.0) -> UncertaintyResult:
        """Run parameter ensemble uncertainty analysis."""
        
        # Generate parameter samples
        param_samples = self._generate_parameter_samples()
        
        # Run forecast for each sample
        eol_samples = []
        soh_samples = []
        
        for params in param_samples:
            model_sample = PhysicsDegradationModel(params)
            eol_info, soh_forecast = self._run_sample_forecast(model_sample, forecast_years)
            eol_samples.append(eol_info['years_to_eol'])
            soh_samples.append(soh_forecast)
        
        eol_samples = np.array(eol_samples)
        soh_samples = np.array(soh_samples)  # Shape: (n_samples, n_forecast_points)
        
        # Calculate percentiles
        eol_percentiles = self._calculate_eol_percentiles(eol_samples)
        soh_bands = self._calculate_soh_bands(soh_samples)
        
        # Generate forecast dates
        n_days = int(forecast_years * 365.25)
        forecast_dates = pd.date_range(
            start=pd.Timestamp.now() + pd.Timedelta(days=1),
            periods=n_days,
            freq='D'
        ).values
        
        return UncertaintyResult(
            eol_samples=eol_samples,
            eol_percentiles=eol_percentiles,
            soh_bands=soh_bands,
            parameter_samples=param_samples,
            forecast_dates=forecast_dates,
        )
    
    def _generate_parameter_samples(self) -> List[PhysicsParameters]:
        """Generate parameter samples around calibrated values."""
        samples = []
        param_dict = self.calibrated_params.to_dict()
        
        for i in range(self.n_samples):
            sampled_params = {}
            for name, value in param_dict.items():
                # Log-normal perturbation for scale parameters, normal for others
                if name in ['A_cal', 'A_cyc', 'Ea_cal', 'Ea_cyc']:
                    # Log-normal for positive scale parameters
                    log_val = np.log(max(value, 1e-10))
                    log_std = self.parameter_perturbation
                    sampled_log = np.random.normal(log_val, log_std)
                    sampled_params[name] = float(np.exp(sampled_log))
                else:
                    # Normal for shape/stress parameters
                    std = abs(value) * self.parameter_perturbation + 1e-6
                    sampled = np.random.normal(value, std)
                    # Clip to bounds
                    from utils.constants import PARAM_BOUNDS
                    if name in PARAM_BOUNDS:
                        low, high = PARAM_BOUNDS[name]
                        sampled = np.clip(sampled, low, high)
                    sampled_params[name] = float(sampled)
            
            samples.append(PhysicsParameters(**sampled_params))
        
        return samples
    
    def _run_sample_forecast(self, 
                             model: PhysicsDegradationModel, 
                             forecast_years: float) -> Tuple[Dict, np.ndarray]:
        """Run forecast for a single parameter sample."""
        # Prepare historical data
        hist_data = self._prepare_historical_data()
        soh0 = float(hist_data['soh_pct'][-1]) if len(hist_data['soh_pct']) > 0 else 100.0
        
        # Future time points
        n_days = int(forecast_years * 365.25)
        t0 = hist_data['time_hours'][-1] if len(hist_data['time_hours']) > 0 else 0
        future_time_hours = np.arange(t0 + 24, t0 + 24 + n_days * 24, 24.0)
        
        # Use median operating conditions (middle scenario)
        stats = self._calculate_operating_stats(hist_data)
        
        temp = np.full(n_days, stats['temp_median'])
        soc = np.full(n_days, stats['soc_median'])
        dod = np.full(n_days, stats['dod_median'])
        efc = np.full(n_days, stats['efc_median'])
        crate = np.full(n_days, stats['crate_median'])
        tmax = np.full(n_days, stats['tmax_median'])
        tmin = np.full(n_days, stats['tmin_median'])
        
        # Run model
        soh_full, _, _ = model.predict_soh(
            time_hours=future_time_hours,
            temperature=temp,
            soc=soc,
            efc=efc,
            dod=dod,
            c_rate=crate,
            tmax=tmax,
            tmin=tmin,
            soh0=soh0
        )
        
        eol_info = predict_eol(soh_full, future_time_hours, EOL_SOH_THRESHOLD)
        
        return eol_info, soh_full
    
    def _prepare_historical_data(self) -> Dict[str, np.ndarray]:
        """Prepare historical data arrays."""
        df = self.field_data.sort_values('time_parsed').reset_index(drop=True)
        
        t0 = df['time_parsed'].min()
        time_hours = (df['time_parsed'] - t0).dt.total_seconds() / 3600.0
        
        return {
            'time_hours': time_hours.values,
            'soh_pct': pd.to_numeric(df['soh_pct'], errors='coerce').values,
            'temperature': pd.to_numeric(df['temperature'], errors='coerce').values,
            'SOC_pct': pd.to_numeric(df['SOC_pct'], errors='coerce').values,
            'Avg_DoD_pct': pd.to_numeric(df['Avg_DoD_pct'], errors='coerce').values,
            'EFC_per_day': pd.to_numeric(df['EFC_per_day'], errors='coerce').values,
            'Cycles_per_day': pd.to_numeric(df['Cycles_per_day'], errors='coerce').values,
            'C_rate': pd.to_numeric(df['C_rate'], errors='coerce').values,
            'Tmax': pd.to_numeric(df['Tmax'], errors='coerce').values,
            'Tmin': pd.to_numeric(df['Tmin'], errors='coerce').values,
        }
    
    def _calculate_operating_stats(self, hist_data: Dict) -> Dict[str, float]:
        """Calculate median operating statistics."""
        def safe_median(arr):
            valid = arr[~np.isnan(arr)]
            return np.median(valid) if len(valid) > 0 else 0.0
        
        return {
            'temp_median': safe_median(hist_data['temperature']),
            'soc_median': safe_median(hist_data['SOC_pct']),
            'dod_median': safe_median(hist_data['Avg_DoD_pct']),
            'efc_median': safe_median(hist_data['EFC_per_day']),
            'crate_median': safe_median(hist_data['C_rate']),
            'tmax_median': safe_median(hist_data['Tmax']),
            'tmin_median': safe_median(hist_data['Tmin']),
        }
    
    def _calculate_eol_percentiles(self, eol_samples: np.ndarray) -> Dict[str, float]:
        """Calculate EOL percentiles."""
        valid = eol_samples[np.isfinite(eol_samples)]
        if len(valid) == 0:
            return {f"P{p}": np.inf for p in [5, 10, 25, 50, 75, 90, 95]}
        
        return {
            "P5": float(np.percentile(valid, 5)),
            "P10": float(np.percentile(valid, 10)),
            "P25": float(np.percentile(valid, 25)),
            "P50": float(np.percentile(valid, 50)),
            "P75": float(np.percentile(valid, 75)),
            "P90": float(np.percentile(valid, 90)),
            "P95": float(np.percentile(valid, 95)),
        }
    
    def _calculate_soh_bands(self, soh_samples: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate SOH uncertainty bands (percentiles over time)."""
        n_samples, n_time = soh_samples.shape
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        bands = {}
        
        for p in percentiles:
            bands[f"P{p}"] = np.percentile(soh_samples, p, axis=0)
        
        return bands


def run_uncertainty_analysis(model: PhysicsDegradationModel,
                             field_data: pd.DataFrame,
                             oem_data: Optional[pd.DataFrame] = None,
                             forecast_years: float = 10.0,
                             n_samples: int = 100) -> UncertaintyResult:
    """Convenience function to run uncertainty analysis."""
    analyzer = UncertaintyAnalyzer(model, field_data, oem_data, n_samples)
    return analyzer.run_uncertainty_analysis(forecast_years)