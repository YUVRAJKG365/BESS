import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings

from utils.constants import T_REF, T_K_REF, EOL_SOH_THRESHOLD, PARAM_BOUNDS
from utils.helpers import safe_float


@dataclass
class PhysicsParameters:
    """Container for physics model parameters."""
    A_cal: float = 1.0e-3
    A_cyc: float = 1.0e-3
    Ea_cal: float = 4.0e4  # J/mol
    Ea_cyc: float = 4.0e4  # J/mol
    alpha: float = 0.5
    beta: float = 1.0
    k_SOC_high: float = 1.0
    k_SOC_reward: float = 0.5
    k_SOC_low: float = 1.0
    gamma_DoD: float = 1.0
    k_C: float = 1.0
    k_Tmax: float = 1.0
    k_Tmin: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert to array for optimization."""
        return np.array([
            self.A_cal, self.A_cyc, self.Ea_cal, self.Ea_cyc,
            self.alpha, self.beta, self.k_SOC_high, self.k_SOC_reward,
            self.k_SOC_low, self.gamma_DoD, self.k_C, self.k_Tmax, self.k_Tmin
        ])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "A_cal": self.A_cal,
            "A_cyc": self.A_cyc,
            "Ea_cal": self.Ea_cal,
            "Ea_cyc": self.Ea_cyc,
            "alpha": self.alpha,
            "beta": self.beta,
            "k_SOC_high": self.k_SOC_high,
            "k_SOC_reward": self.k_SOC_reward,
            "k_SOC_low": self.k_SOC_low,
            "gamma_DoD": self.gamma_DoD,
            "k_C": self.k_C,
            "k_Tmax": self.k_Tmax,
            "k_Tmin": self.k_Tmin,
        }
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PhysicsParameters':
        """Create from array."""
        return cls(*arr)


class PhysicsDegradationModel:
    """
    Physics-based BESS degradation model following V11 methodology.
    
    Implements:
    - Calendar aging with Arrhenius temperature dependence
    - Cycle aging with DoD, SOC, C-rate stress factors
    - Power-law time and throughput dependence
    - High/low temperature stress (Tmax/Tmin)
    """
    
    R = 8.314462618  # Universal gas constant J/(mol·K)
    
    def __init__(self, params: Optional[PhysicsParameters] = None):
        self.params = params or PhysicsParameters()
        self.calibration_history: List[Dict] = []
        self.is_calibrated = False
    
    def _arrhenius_factor(self, Ea: float, T_C: float) -> float:
        """Calculate Arrhenius temperature acceleration factor."""
        T_K = T_C + 273.15
        return np.exp(-Ea / self.R * (1.0 / T_K - 1.0 / T_K_REF))
    
    def _soc_stress_factor(self, soc_pct: float) -> float:
        """
        Calculate SOC stress factor.
        High SOC (>80%): accelerated degradation
        Low SOC (<20%): accelerated degradation
        Mid SOC: baseline
        """
        if soc_pct >= 80:
            # Exponential stress for high SOC
            return np.exp(self.params.k_SOC_high * (soc_pct - 80) / 20.0)
        elif soc_pct <= 20:
            # Exponential stress for low SOC
            return np.exp(self.params.k_SOC_low * (20 - soc_pct) / 20.0)
        else:
            # Reward zone - reduced stress
            return np.exp(-self.params.k_SOC_reward * (soc_pct - 50) / 30.0)
    
    def _dod_stress_factor(self, dod_pct: float) -> float:
        """Calculate DoD stress factor using power law."""
        dod_frac = dod_pct / 100.0
        return dod_frac ** self.params.gamma_DoD
    
    def _crate_stress_factor(self, c_rate: float) -> float:
        """Calculate C-rate stress factor."""
        return np.exp(self.params.k_C * (c_rate - 1.0))
    
    def _tmax_stress_factor(self, tmax_c: float) -> float:
        """High temperature stress factor (Peaks Over Threshold)."""
        if tmax_c > 35:
            return np.exp(self.params.k_Tmax * (tmax_c - 35) / 10.0)
        return 1.0
    
    def _tmin_stress_factor(self, tmin_c: float) -> float:
        """Low temperature / cold cycle stress factor."""
        if tmin_c < 0:
            return np.exp(self.params.k_Tmin * (0 - tmin_c) / 10.0)
        return 1.0
    
    def calendar_degradation(self, 
                            time_hours: np.ndarray,
                            temperature: np.ndarray,
                            soc: np.ndarray) -> np.ndarray:
        """
        Calculate calendar aging degradation.
        
        ΔSOH_cal = A_cal * t^alpha * exp(-Ea_cal/R*(1/T-1/Tref)) * f_SOC(SOC)
        """
        # Time in years
        time_years = time_hours / (365.25 * 24)
        
        # Time power law
        time_factor = time_years ** self.params.alpha
        
        # Temperature factor (Arrhenius)
        temp_factor = np.array([self._arrhenius_factor(self.params.Ea_cal, t) for t in temperature])
        
        # SOC stress factor
        soc_factor = np.array([self._soc_stress_factor(s) for s in soc])
        
        # Total calendar degradation
        deg = self.params.A_cal * time_factor * temp_factor * soc_factor
        
        return deg
    
    def cycle_degradation(self,
                         efc: np.ndarray,
                         dod: np.ndarray,
                         temperature: np.ndarray,
                         soc: np.ndarray,
                         c_rate: np.ndarray,
                         tmax: np.ndarray,
                         tmin: np.ndarray) -> np.ndarray:
        """
        Calculate cycle aging degradation.
        
        ΔSOH_cyc = A_cyc * (EFC)^beta * f_DoD(DoD) * f_T(T) * f_SOC(SOC) * f_C(C) * f_Tmax(Tmax) * f_Tmin(Tmin)
        """
        # EFC throughput power law
        efc_factor = efc ** self.params.beta
        
        # DoD stress
        dod_factor = np.array([self._dod_stress_factor(d) for d in dod])
        
        # Temperature factor (Arrhenius for cycling)
        temp_factor = np.array([self._arrhenius_factor(self.params.Ea_cyc, t) for t in temperature])
        
        # SOC stress
        soc_factor = np.array([self._soc_stress_factor(s) for s in soc])
        
        # C-rate stress
        crate_factor = np.array([self._crate_stress_factor(c) for c in c_rate])
        
        # Tmax stress
        tmax_factor = np.array([self._tmax_stress_factor(t) for t in tmax])
        
        # Tmin stress
        tmin_factor = np.array([self._tmin_stress_factor(t) for t in tmin])
        
        # Total cycle degradation
        deg = (self.params.A_cyc * efc_factor * dod_factor * temp_factor * 
               soc_factor * crate_factor * tmax_factor * tmin_factor)
        
        return deg
    
    def predict_soh(self,
                    time_hours: np.ndarray,
                    temperature: np.ndarray,
                    soc: np.ndarray,
                    efc: np.ndarray,
                    dod: np.ndarray,
                    c_rate: np.ndarray,
                    tmax: np.ndarray,
                    tmin: np.ndarray,
                    soh0: float = 100.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict SOH degradation over time.
        
        Returns: (total_soh, calendar_loss, cycle_loss)
        """
        # Calendar degradation (cumulative)
        cal_deg = self.calendar_degradation(time_hours, temperature, soc)
        
        # Cycle degradation (cumulative)
        cyc_deg = self.cycle_degradation(efc, dod, temperature, soc, c_rate, tmax, tmin)
        
        # Cumulative sums
        cal_cum = np.cumsum(cal_deg)
        cyc_cum = np.cumsum(cyc_deg)
        
        total_loss = cal_cum + cyc_cum
        soh = np.clip(soh0 - total_loss, 0.0, 100.0)
        
        return soh, cal_cum, cyc_cum
    
    def predict_single_step(self,
                           dt_hours: float,
                           temperature: float,
                           soc: float,
                           efc: float,
                           dod: float,
                           c_rate: float,
                           tmax: float,
                           tmin: float,
                           prev_soh: float,
                           prev_cal_loss: float,
                           prev_cyc_loss: float) -> Tuple[float, float, float]:
        """Predict SOH for a single time step."""
        cal = self.calendar_degradation(np.array([dt_hours]), 
                                        np.array([temperature]), 
                                        np.array([soc]))[0]
        cyc = self.cycle_degradation(np.array([efc]), 
                                     np.array([dod]), 
                                     np.array([temperature]),
                                     np.array([soc]),
                                     np.array([c_rate]),
                                     np.array([tmax]),
                                     np.array([tmin]))[0]
        
        new_cal = prev_cal_loss + cal
        new_cyc = prev_cyc_loss + cyc
        new_soh = prev_soh - cal - cyc
        
        return new_soh, new_cal, new_cyc


def detect_soh_steps(soh: np.ndarray, time: np.ndarray, 
                     threshold_pct: float = 1.0) -> List[int]:
    """Detect step events in SOH time series (sudden drops > threshold)."""
    steps = []
    if len(soh) < 3:
        return steps
    
    diffs = np.diff(soh)
    # Negative jumps (degradation steps)
    step_indices = np.where(diffs < -threshold_pct)[0]
    
    for idx in step_indices:
        # Check if it's a real step (not just noise)
        if idx + 2 < len(soh):
            before = np.mean(soh[max(0, idx-2):idx])
            after = np.mean(soh[idx+1:idx+3])
            if before - after > threshold_pct:
                steps.append(idx)
    
    return steps


def smooth_soh_target(soh: np.ndarray, time: np.ndarray, 
                      window: int = 5) -> np.ndarray:
    """Generate smooth SOH target using rolling median filter."""
    if len(soh) < window:
        return soh.copy()
    
    # Use rolling median for robust smoothing
    smooth = pd.Series(soh).rolling(window=window, center=True, min_periods=1).median().values
    return smooth


def prepare_oem_baseline(oem_df: Any, 
                         time_col: str = 'time', 
                         soh_col: str = 'soh_pct') -> Tuple[np.ndarray, np.ndarray]:
    """Prepare OEM baseline data for calibration."""
    if hasattr(oem_df, 'processed_time') and hasattr(oem_df, 'processed_soh'):
        if len(oem_df.processed_time) > 0 and len(oem_df.processed_soh) > 0:
            return np.asarray(oem_df.processed_time), np.asarray(oem_df.processed_soh)
    if hasattr(oem_df, 'raw_data') and isinstance(oem_df.raw_data, pd.DataFrame):
        oem_df = oem_df.raw_data

    # Detect time and soh cols if default not in df
    if time_col not in oem_df.columns:
        for c in ['Day', 'day', 'time', 'Year', 'year', 'date', 'Date']:
            if c in oem_df.columns:
                time_col = c
                break
    if soh_col not in oem_df.columns:
        for c in ['OEM_SOH_pct', 'oem_soh_pct', 'SOH_pct', 'soh_pct', 'SOH', 'soh']:
            if c in oem_df.columns:
                soh_col = c
                break

    oem_time = pd.to_numeric(oem_df[time_col], errors='coerce').values
    oem_soh = pd.to_numeric(oem_df[soh_col], errors='coerce').values
    
    # Remove NaN
    valid = ~np.isnan(oem_time) & ~np.isnan(oem_soh)
    oem_time = oem_time[valid]
    oem_soh = oem_soh[valid]
    
    # Ensure sorted
    sort_idx = np.argsort(oem_time)
    return oem_time[sort_idx], oem_soh[sort_idx]


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate model performance metrics."""
    valid = ~np.isnan(observed) & ~np.isnan(predicted)
    if not np.any(valid):
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan}
    
    obs = observed[valid]
    pred = predicted[valid]
    
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae = np.mean(np.abs(obs - pred))
    
    # R²
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    # MAPE
    mape = np.mean(np.abs((obs - pred) / obs)) * 100 if np.all(obs != 0) else np.nan
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
    }


def predict_eol(soh_forecast: np.ndarray, time_forecast: np.ndarray, 
                eol_threshold: float = EOL_SOH_THRESHOLD) -> Dict[str, Any]:
    """Predict End of Life from forecast."""
    # Find first time SOH crosses threshold
    below = soh_forecast <= eol_threshold
    
    if np.any(below):
        eol_idx = np.argmax(below)
        eol_time = time_forecast[eol_idx]
        eol_soh = soh_forecast[eol_idx]
        years_to_eol = eol_time / (365.25 * 24) if eol_time > 0 else np.inf
        
        return {
            "eol_reached": True,
            "eol_time_hours": float(eol_time),
            "eol_soh_pct": float(eol_soh),
            "years_to_eol": float(years_to_eol),
            "eol_date": None,  # Will be set by caller if reference date provided
        }
    else:
        # Extrapolate
        if len(soh_forecast) > 1:
            # Linear extrapolation
            slope = (soh_forecast[-1] - soh_forecast[0]) / (time_forecast[-1] - time_forecast[0])
            if slope < 0:
                extra_hours = (eol_threshold - soh_forecast[-1]) / slope + time_forecast[-1]
                return {
                    "eol_reached": False,
                    "eol_time_hours": float(extra_hours),
                    "eol_soh_pct": eol_threshold,
                    "years_to_eol": float(extra_hours / (365.25 * 24)),
                    "eol_date": None,
                }
        
        return {
            "eol_reached": False,
            "eol_time_hours": np.inf,
            "eol_soh_pct": float(soh_forecast[-1]),
            "years_to_eol": np.inf,
            "eol_date": None,
        }