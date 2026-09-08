import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from backend.physics_model import (
    PhysicsParameters, PhysicsDegradationModel, predict_eol, calculate_metrics
)
from utils.constants import EOL_SOH_THRESHOLD
from utils.helpers import safe_float


@dataclass
class ForecastResult:
    """Container for forecast results."""
    historical_dates: np.ndarray
    historical_soh: np.ndarray
    historical_fit: np.ndarray
    forecast_dates: np.ndarray
    forecast_soh: np.ndarray
    calendar_loss_historical: np.ndarray
    cycle_loss_historical: np.ndarray
    calendar_loss_forecast: np.ndarray
    cycle_loss_forecast: np.ndarray
    eol_info: Dict[str, Any]
    metrics: Dict[str, float]
    parameters: PhysicsParameters
    oem_forecast: Optional[np.ndarray] = None
    oem_dates: Optional[np.ndarray] = None
    soh_uncertainty_lower: Optional[np.ndarray] = None
    soh_uncertainty_upper: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)


class PhysicsForecaster:
    """
    Physics-based forecasting engine.
    
    Generates forecasts based on calibrated model parameters and future operating scenarios.
    """
    
    def __init__(self, 
                 model: PhysicsDegradationModel,
                 field_data: pd.DataFrame,
                 oem_data: Optional[pd.DataFrame] = None):
        self.model = model
        self.field_data = field_data.copy()
        self.oem_data = oem_data
        self.reference_date = field_data['time_parsed'].max() if 'time_parsed' in field_data.columns else datetime.now()
    
    def forecast(self,
                 forecast_years: float = 10.0,
                 scenario: str = "middle",
                 n_scenarios: int = 100) -> ForecastResult:
        """
        Run forecast for specified scenario.
        
        Args:
            forecast_years: Years to forecast into future
            scenario: "best", "middle", "worst" or "custom"
            n_scenarios: Number of Monte Carlo scenarios for uncertainty
            
        Returns:
            ForecastResult with historical fit and future forecast
        """
        # Prepare historical data
        hist_data = self._prepare_historical_data()
        
        # Generate future operating conditions based on scenario
        future_data = self._generate_future_conditions(forecast_years, scenario)
        
        # Combine historical and future
        combined = self._combine_data(hist_data, future_data)
        
        # Run model prediction
        soh_full, cal_loss, cyc_loss = self.model.predict_soh(
            time_hours=combined['time_hours'],
            temperature=combined['temperature'],
            soc=combined['SOC_pct'],
            efc=combined['EFC_per_day'],
            dod=combined['Avg_DoD_pct'],
            c_rate=combined['C_rate'],
            tmax=combined['Tmax'],
            tmin=combined['Tmin'],
            soh0=100.0
        )
        
        # Split historical and forecast
        n_hist = len(hist_data['time_hours'])
        soh_hist = soh_full[:n_hist]
        soh_forecast = soh_full[n_hist:]
        cal_hist = cal_loss[:n_hist]
        cal_forecast = cal_loss[n_hist:]
        cyc_hist = cyc_loss[:n_hist]
        cyc_forecast = cyc_loss[n_hist:]
        
        # Generate dates
        hist_dates = hist_data['dates']
        forecast_dates = future_data['dates']
        
        # Calculate EOL
        eol_info = predict_eol(soh_forecast, future_data['time_hours'], EOL_SOH_THRESHOLD)
        
        # Calculate metrics
        metrics = calculate_metrics(hist_data['soh_pct'], soh_hist)
        
        # OEM forecast if available
        oem_forecast, oem_dates = None, None
        if self.oem_data is not None:
            oem_forecast, oem_dates = self._generate_oem_forecast(forecast_years)
        
        return ForecastResult(
            historical_dates=hist_dates,
            historical_soh=hist_data['soh_pct'],
            historical_fit=soh_hist,
            forecast_dates=forecast_dates,
            forecast_soh=soh_forecast,
            calendar_loss_historical=cal_hist,
            cycle_loss_historical=cyc_hist,
            calendar_loss_forecast=cal_forecast,
            cycle_loss_forecast=cyc_forecast,
            eol_info=eol_info,
            metrics=metrics,
            parameters=self.model.params,
            oem_forecast=oem_forecast,
            oem_dates=oem_dates,
            warnings=self._generate_warnings(soh_forecast, eol_info),
        )
    
    def _prepare_historical_data(self) -> Dict[str, np.ndarray]:
        """Prepare historical data arrays."""
        df = self.field_data.sort_values('time_parsed').reset_index(drop=True)
        
        # Time in hours from start
        n = len(df)
        if 'time_parsed' in df.columns:
            t_dt = pd.to_datetime(df['time_parsed'], errors='coerce')
            t0 = t_dt.iloc[0] if len(t_dt) > 0 and pd.notna(t_dt.iloc[0]) else pd.Timestamp('2024-01-01')
            time_hours = (t_dt - t0).dt.total_seconds() / 3600.0
            dates = t_dt.values
        else:
            time_hours = pd.Series(np.arange(n, dtype=float) * 24.0)
            dates = pd.date_range('2024-01-01', periods=n, freq='D').values

        def _get_col(col_name, default_val):
            if col_name in df.columns:
                s = pd.to_numeric(df[col_name], errors='coerce')
                if isinstance(default_val, np.ndarray):
                    return np.where(s.notna().values, s.values, default_val)
                return s.fillna(default_val).values
            if isinstance(default_val, np.ndarray):
                return default_val.copy()
            return np.full(n, default_val, dtype=float)

        efc = _get_col('EFC_per_day', 1.0)
        temp = _get_col('temperature', 25.0)
        
        return {
            'time_hours': time_hours.values,
            'dates': dates,
            'soh_pct': _get_col('soh_pct', 100.0),
            'temperature': temp,
            'SOC_pct': _get_col('SOC_pct', 50.0),
            'Avg_DoD_pct': _get_col('Avg_DoD_pct', 95.0),
            'EFC_per_day': efc,
            'Cycles_per_day': _get_col('Cycles_per_day', efc),
            'C_rate': _get_col('C_rate', 0.25),
            'Tmax': _get_col('Tmax', temp),
            'Tmin': _get_col('Tmin', temp),
        }
    
    def _generate_future_conditions(self, 
                                    forecast_years: float, 
                                    scenario: str) -> Dict[str, np.ndarray]:
        """Generate future operating conditions based on historical statistics."""
        hist = self._prepare_historical_data()
        
        # Calculate historical statistics
        stats = self._calculate_operating_stats(hist)
        
        # Define scenario multipliers
        scenario_factors = {
            'best': {
                'temp_factor': 0.9,
                'soc_factor': 0.9,  # More moderate SOC
                'dod_factor': 0.8,
                'efc_factor': 0.7,
                'crate_factor': 0.8,
            },
            'middle': {
                'temp_factor': 1.0,
                'soc_factor': 1.0,
                'dod_factor': 1.0,
                'efc_factor': 1.0,
                'crate_factor': 1.0,
            },
            'worst': {
                'temp_factor': 1.2,
                'soc_factor': 1.15,  # Higher SOC stress
                'dod_factor': 1.3,
                'efc_factor': 1.5,
                'crate_factor': 1.3,
            },
        }
        
        factors = scenario_factors.get(scenario, scenario_factors['middle'])
        
        # Future time points (daily resolution)
        n_days = int(forecast_years * 365.25)
        future_time_hours = np.arange(
            hist['time_hours'][-1] + 24,  # Start next day
            hist['time_hours'][-1] + 24 + n_days * 24,
            24.0
        )
        
        # Generate future dates
        future_dates = pd.date_range(
            start=self.reference_date + timedelta(days=1),
            periods=n_days,
            freq='D'
        )
        
        # Apply scenario factors to historical statistics
        temp_mean = stats['temp_mean'] * factors['temp_factor']
        temp_std = stats['temp_std']
        
        # Temperature with seasonal variation
        day_of_year = np.arange(n_days) % 365.25
        seasonal_temp = 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        temperature = temp_mean + seasonal_temp + np.random.normal(0, temp_std, n_days)
        
        # SOC - more moderate in best case
        soc_mean = np.clip(stats['soc_mean'] * factors['soc_factor'], 20, 80)
        soc = np.clip(np.random.normal(soc_mean, stats['soc_std'], n_days), 0, 100)
        
        # DoD
        dod = np.full(n_days, np.clip(stats['dod_mean'] * factors['dod_factor'], 5, 100))
        
        # EFC per day
        efc = np.full(n_days, np.maximum(stats['efc_mean'] * factors['efc_factor'], 0.1))
        
        # C-rate
        crate = np.full(n_days, np.clip(stats['crate_mean'] * factors['crate_factor'], 0.1, 5.0))
        
        # Tmax and Tmin
        tmax = temperature + np.abs(np.random.normal(5, 2, n_days))
        tmin = temperature - np.abs(np.random.normal(5, 2, n_days))
        
        return {
            'time_hours': future_time_hours,
            'dates': future_dates.values,
            'temperature': temperature,
            'SOC_pct': soc,
            'Avg_DoD_pct': dod,
            'EFC_per_day': efc,
            'Cycles_per_day': efc,  # Assume 1 EFC = 1 cycle
            'C_rate': crate,
            'Tmax': tmax,
            'Tmin': tmin,
        }
    
    def _calculate_operating_stats(self, hist: Dict) -> Dict[str, float]:
        """Calculate operating statistics from historical data."""
        return {
            'temp_mean': float(np.nanmean(hist['temperature'])),
            'temp_std': float(np.nanstd(hist['temperature'])),
            'soc_mean': float(np.nanmean(hist['SOC_pct'])),
            'soc_std': float(np.nanstd(hist['SOC_pct'])),
            'dod_mean': float(np.nanmean(hist['Avg_DoD_pct'])),
            'dod_std': float(np.nanstd(hist['Avg_DoD_pct'])),
            'efc_mean': float(np.nanmean(hist['EFC_per_day'])),
            'efc_std': float(np.nanstd(hist['EFC_per_day'])),
            'crate_mean': float(np.nanmean(hist['C_rate'])),
            'crate_std': float(np.nanstd(hist['C_rate'])),
        }
    
    def _combine_data(self, hist: Dict, future: Dict) -> Dict[str, np.ndarray]:
        """Combine historical and future data arrays."""
        keys = ['time_hours', 'temperature', 'SOC_pct', 'Avg_DoD_pct', 
                'EFC_per_day', 'Cycles_per_day', 'C_rate', 'Tmax', 'Tmin']
        
        combined = {}
        for key in keys:
            h_arr = np.atleast_1d(hist[key])
            f_arr = np.atleast_1d(future[key])
            combined[key] = np.concatenate([h_arr, f_arr])
        
        return combined
    
    def _generate_oem_forecast(self, forecast_years: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate OEM baseline forecast."""
        if self.oem_data is None:
            return None, None
        
        if hasattr(self.oem_data, 'processed_time') and hasattr(self.oem_data, 'processed_soh'):
            oem_time = np.asarray(self.oem_data.processed_time)
            oem_soh = np.asarray(self.oem_data.processed_soh)
        elif isinstance(self.oem_data, dict):
            oem_time = np.asarray(self.oem_data.get('time'))
            oem_soh = np.asarray(self.oem_data.get('soh'))
        elif isinstance(self.oem_data, (tuple, list)) and len(self.oem_data) == 2:
            oem_time = np.asarray(self.oem_data[0])
            oem_soh = np.asarray(self.oem_data[1])
        else:
            return None, None
        
        if oem_time is None or oem_soh is None or len(oem_time) == 0:
            return None, None
        
        # Simple exponential decay fit for OEM
        # log(SOH) = log(100) - k*t
        valid = ~np.isnan(oem_soh) & (oem_soh > 0)
        if np.sum(valid) < 3:
            return None, None
        
        t_valid = oem_time[valid]
        soh_valid = oem_soh[valid]
        
        # Fit exponential
        log_soh = np.log(soh_valid / 100.0)
        k = -np.polyfit(t_valid, log_soh, 1)[0] if len(t_valid) > 1 else 0.0001
        
        # Future OEM forecast
        n_days = int(forecast_years * 365.25)
        future_oem_time = np.arange(
            oem_time[-1] + 24,
            oem_time[-1] + 24 + n_days * 24,
            24.0
        )
        future_oem_soh = 100 * np.exp(-k * future_oem_time)
        
        future_oem_dates = pd.date_range(
            start=self.reference_date + timedelta(days=1),
            periods=n_days,
            freq='D'
        )
        
        return future_oem_soh, future_oem_dates.values
    
    def _generate_warnings(self, soh_forecast: np.ndarray, eol_info: Dict) -> List[str]:
        """Generate forecast warnings."""
        warnings = []
        
        if eol_info['years_to_eol'] < 1:
            warnings.append("EOL predicted within 1 year - immediate attention required")
        elif eol_info['years_to_eol'] < 5:
            warnings.append(f"EOL predicted in {eol_info['years_to_eol']:.1f} years - plan replacement")
        
        final_soh = soh_forecast[-1]
        if final_soh < 50:
            warnings.append(f"Final forecast SOH very low: {final_soh:.1f}%")
        
        return warnings


def run_forecast(model: PhysicsDegradationModel,
                 field_data: pd.DataFrame,
                 oem_data: Optional[pd.DataFrame] = None,
                 forecast_years: float = 10.0,
                 scenario: str = "middle") -> ForecastResult:
    """Convenience function to run forecast."""
    forecaster = PhysicsForecaster(model, field_data, oem_data)
    return forecaster.forecast(forecast_years, scenario)