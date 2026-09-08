import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from backend.physics_model import PhysicsDegradationModel, PhysicsParameters, predict_eol, calculate_metrics
from utils.constants import EOL_SOH_THRESHOLD


@dataclass
class ScenarioResult:
    """Container for a single scenario result."""
    scenario_name: str
    scenario_type: str  # "best", "middle", "worst"
    
    # Operating conditions
    EFC_per_day: float
    Avg_SOC_pct: float
    Avg_DoD_pct: float
    Mean_Tavg_C: float
    Max_Tmax_C: float
    Min_Tmin_C: float
    Cycles_per_day: float
    Charge_Duration_hr: float
    Discharge_Duration_hr: float
    C_rate: float
    
    # Forecast results
    EOL_year: float
    EOL_date: Optional[str]
    Final_SOH_pct: float
    calendar_loss: float
    cycle_loss: float
    total_loss: float
    
    # Full forecast data
    forecast_dates: np.ndarray
    forecast_soh: np.ndarray
    calendar_loss_ts: np.ndarray
    cycle_loss_ts: np.ndarray
    
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type,
            "EFC_per_day": self.EFC_per_day,
            "Avg_SOC_pct": self.Avg_SOC_pct,
            "Avg_DoD_pct": self.Avg_DoD_pct,
            "Mean_Tavg_C": self.Mean_Tavg_C,
            "Max_Tmax_C": self.Max_Tmax_C,
            "Min_Tmin_C": self.Min_Tmin_C,
            "Cycles_per_day": self.Cycles_per_day,
            "Charge_Duration_hr": self.Charge_Duration_hr,
            "Discharge_Duration_hr": self.Discharge_Duration_hr,
            "C_rate": self.C_rate,
            "EOL_year": self.EOL_year,
            "EOL_date": self.EOL_date,
            "Final_SOH_pct": self.Final_SOH_pct,
            "calendar_loss": self.calendar_loss,
            "cycle_loss": self.cycle_loss,
            "total_loss": self.total_loss,
            "description": self.description,
        }


class ScenarioAnalyzer:
    """
    Generates best/middle/worst case scenarios from field data statistics.
    
    Uses actual observed distributions/quantiles to define scenarios.
    """
    
    def __init__(self, model: PhysicsDegradationModel, field_data: pd.DataFrame):
        self.model = model
        self.field_data = field_data.copy()
        self.operating_stats = self._calculate_operating_statistics()
    
    def _calculate_operating_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive operating statistics from field data."""
        df = self.field_data
        
        # Ensure numeric
        numeric_cols = {
            'temperature': 'temperature',
            'SOC_pct': 'SOC_pct',
            'Avg_DoD_pct': 'Avg_DoD_pct',
            'EFC_per_day': 'EFC_per_day',
            'Cycles_per_day': 'Cycles_per_day',
            'Charge_Duration_hr': 'Charge_Duration_hr',
            'Discharge_Duration_hr': 'Discharge_Duration_hr',
            'C_rate': 'C_rate',
            'Tmax': 'Tmax',
            'Tmin': 'Tmin',
        }
        
        stats = {}
        for key, col in numeric_cols.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(vals) > 0:
                    stats[f"{key}_mean"] = float(vals.mean())
                    stats[f"{key}_std"] = float(vals.std())
                    stats[f"{key}_median"] = float(vals.median())
                    stats[f"{key}_q25"] = float(vals.quantile(0.25))
                    stats[f"{key}_q75"] = float(vals.quantile(0.75))
                    stats[f"{key}_min"] = float(vals.min())
                    stats[f"{key}_max"] = float(vals.max())
                else:
                    stats[f"{key}_mean"] = self._get_default(key)
                    stats[f"{key}_std"] = 0.0
            else:
                stats[f"{key}_mean"] = self._get_default(key)
                stats[f"{key}_std"] = 0.0
        
        return stats
    
    def _get_default(self, key: str) -> float:
        """Get default value for missing statistics."""
        defaults = {
            'temperature': 25.0,
            'SOC_pct': 50.0,
            'Avg_DoD_pct': 50.0,
            'EFC_per_day': 1.0,
            'Cycles_per_day': 1.0,
            'Charge_Duration_hr': 2.0,
            'Discharge_Duration_hr': 2.0,
            'C_rate': 0.5,
            'Tmax': 35.0,
            'Tmin': 15.0,
        }
        return defaults.get(key, 0.0)
    
    def generate_scenarios(self, forecast_years: float = 10.0) -> Dict[str, ScenarioResult]:
        """
        Generate best, middle, and worst case scenarios.
        
        Returns:
            Dictionary with scenario results keyed by scenario type
        """
        scenarios = {}
        
        # Define scenario parameters using quantiles
        scenario_defs = {
            'best': {
                'temp_factor': 'q25',  # Cooler temperatures
                'soc_factor': 'q25',   # Lower SOC (but not too low)
                'dod_factor': 'q25',   # Lower DoD
                'efc_factor': 'q25',   # Lower cycling
                'crate_factor': 'q25', # Lower C-rate
                'description': "Lower degradation stress - favorable operating conditions"
            },
            'middle': {
                'temp_factor': 'median',
                'soc_factor': 'median',
                'dod_factor': 'median',
                'efc_factor': 'median',
                'crate_factor': 'median',
                'description': "Representative/median field operating conditions"
            },
            'worst': {
                'temp_factor': 'q75',  # Hotter temperatures
                'soc_factor': 'q75',   # Higher SOC stress
                'dod_factor': 'q75',   # Higher DoD
                'efc_factor': 'q75',   # Higher cycling
                'crate_factor': 'q75', # Higher C-rate
                'description': "Higher degradation stress - challenging operating conditions"
            },
        }
        
        for scenario_name, defs in scenario_defs.items():
            # Get scenario operating conditions
            conditions = self._get_scenario_conditions(defs)
            
            # Run forecast for this scenario
            scenario_result = self._run_scenario_forecast(
                scenario_name, defs['description'], conditions, forecast_years
            )
            
            scenarios[scenario_name] = scenario_result
        
        return scenarios
    
    def _get_scenario_conditions(self, defs: Dict[str, str]) -> Dict[str, float]:
        """Extract operating conditions for a scenario from statistics."""
        conditions = {}
        
        mapping = {
            'temperature': 'temp_factor',
            'SOC_pct': 'soc_factor',
            'Avg_DoD_pct': 'dod_factor',
            'EFC_per_day': 'efc_factor',
            'C_rate': 'crate_factor',
        }
        
        for key, factor_key in mapping.items():
            factor = defs.get(factor_key, 'median')
            stat_key = f"{key}_{factor}"
            conditions[key] = self.operating_stats.get(stat_key, self.operating_stats.get(f"{key}_mean", 0))
        
        # Derived values
        conditions['Tmax'] = self.operating_stats.get(f"Tmax_{defs['temp_factor']}", 
                                                        conditions['temperature'] + 10)
        conditions['Tmin'] = self.operating_stats.get(f"Tmin_{defs['temp_factor']}", 
                                                        conditions['temperature'] - 10)
        conditions['Cycles_per_day'] = conditions['EFC_per_day']
        conditions['Charge_Duration_hr'] = self.operating_stats.get(
            f"Charge_Duration_hr_{defs['dod_factor']}", 2.0)
        conditions['Discharge_Duration_hr'] = self.operating_stats.get(
            f"Discharge_Duration_hr_{defs['dod_factor']}", 2.0)
        
        return conditions
    
    def _run_scenario_forecast(self,
                               scenario_name: str,
                               description: str,
                               conditions: Dict[str, float],
                               forecast_years: float) -> ScenarioResult:
        """Run forecast for a specific scenario."""
        # Prepare historical data for initial conditions
        hist_data = self._prepare_historical_data()
        soh0 = float(hist_data['soh_pct'][-1]) if len(hist_data['soh_pct']) > 0 else 100.0
        
        # Future time points
        n_days = int(forecast_years * 365.25)
        t0 = hist_data['time_hours'][-1] if len(hist_data['time_hours']) > 0 else 0
        future_time_hours = np.arange(t0 + 24, t0 + 24 + n_days * 24, 24.0)
        future_time_rel = np.arange(24, n_days * 24 + 24, 24.0)
        
        future_dates = pd.date_range(
            start=pd.Timestamp.now() + pd.Timedelta(days=1),
            periods=n_days,
            freq='D'
        ).values
        
        # Build constant conditions for forecast period
        temp = np.full(n_days, conditions['temperature'])
        soc = np.full(n_days, conditions['SOC_pct'])
        dod = np.full(n_days, conditions['Avg_DoD_pct'])
        efc = np.full(n_days, conditions['EFC_per_day'])
        crate = np.full(n_days, conditions['C_rate'])
        tmax = np.full(n_days, conditions['Tmax'])
        tmin = np.full(n_days, conditions['Tmin'])
        
        # Run model
        soh_full, cal_loss, cyc_loss = self.model.predict_soh(
            time_hours=future_time_rel,
            temperature=temp,
            soc=soc,
            efc=efc,
            dod=dod,
            c_rate=crate,
            tmax=tmax,
            tmin=tmin,
            soh0=soh0
        )
        
        # EOL prediction
        eol_info = predict_eol(soh_full, future_time_hours, EOL_SOH_THRESHOLD)
        
        return ScenarioResult(
            scenario_name=scenario_name.capitalize(),
            scenario_type=scenario_name,
            EFC_per_day=conditions['EFC_per_day'],
            Avg_SOC_pct=conditions['SOC_pct'],
            Avg_DoD_pct=conditions['Avg_DoD_pct'],
            Mean_Tavg_C=conditions['temperature'],
            Max_Tmax_C=conditions['Tmax'],
            Min_Tmin_C=conditions['Tmin'],
            Cycles_per_day=conditions['Cycles_per_day'],
            Charge_Duration_hr=conditions['Charge_Duration_hr'],
            Discharge_Duration_hr=conditions['Discharge_Duration_hr'],
            C_rate=conditions['C_rate'],
            EOL_year=eol_info['years_to_eol'],
            EOL_date=None,  # Will be filled by caller with reference date
            Final_SOH_pct=float(soh_full[-1]),
            calendar_loss=float(cal_loss[-1]),
            cycle_loss=float(cyc_loss[-1]),
            total_loss=float(cal_loss[-1] + cyc_loss[-1]),
            forecast_dates=future_dates,
            forecast_soh=soh_full,
            calendar_loss_ts=cal_loss,
            cycle_loss_ts=cyc_loss,
            description=description,
        )
    
    def _prepare_historical_data(self) -> Dict[str, np.ndarray]:
        """Prepare historical data arrays."""
        df = self.field_data.sort_values('time_parsed').reset_index(drop=True)
        
        t0 = df['time_parsed'].min()
        time_hours = (df['time_parsed'] - t0).dt.total_seconds() / 3600.0
        
        return {
            'time_hours': time_hours.values,
            'soh_pct': pd.to_numeric(df['soh_pct'], errors='coerce').values,
        }
    
    def compare_scenarios(self, scenarios: Dict[str, ScenarioResult]) -> pd.DataFrame:
        """Create comparison table for scenarios."""
        rows = []
        for name, s in scenarios.items():
            rows.append({
                "Scenario": s.scenario_name,
                "EFC/day": f"{s.EFC_per_day:.2f}",
                "Avg SOC (%)": f"{s.Avg_SOC_pct:.1f}",
                "Avg DoD (%)": f"{s.Avg_DoD_pct:.1f}",
                "Mean Temp (°C)": f"{s.Mean_Tavg_C:.1f}",
                "Max Temp (°C)": f"{s.Max_Tmax_C:.1f}",
                "Cycles/day": f"{s.Cycles_per_day:.2f}",
                "C-rate": f"{s.C_rate:.2f}",
                "EOL (years)": f"{s.EOL_year:.1f}",
                "Final SOH (%)": f"{s.Final_SOH_pct:.1f}",
                "Cal. Loss": f"{s.calendar_loss:.2f}",
                "Cyc. Loss": f"{s.cycle_loss:.2f}",
            })
        return pd.DataFrame(rows)


def run_scenario_analysis(model: PhysicsDegradationModel,
                          field_data: pd.DataFrame,
                          forecast_years: float = 10.0) -> Dict[str, ScenarioResult]:
    """Convenience function to run scenario analysis."""
    analyzer = ScenarioAnalyzer(model, field_data)
    return analyzer.generate_scenarios(forecast_years)