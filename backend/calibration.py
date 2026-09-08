import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from scipy.optimize import minimize, differential_evolution, least_squares
from scipy import stats
import warnings

from backend.physics_model import (
    PhysicsParameters, PhysicsDegradationModel, 
    smooth_soh_target, detect_soh_steps, prepare_oem_baseline, calculate_metrics
)
from utils.constants import PARAM_BOUNDS, EOL_SOH_THRESHOLD


@dataclass
class CalibrationResult:
    """Result of the calibration process."""
    params: PhysicsParameters
    stage1_result: Optional[Dict] = None
    stage2_result: Optional[Dict] = None
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    param_status: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    model_fit: Optional[np.ndarray] = None


class TwoStageCalibrator:
    """
    Two-stage calibration following V11 methodology.
    
    Stage 1: Global optimization (differential evolution) for rough parameter space exploration
    Stage 2: Local refinement (least squares) for precise parameter estimation
    """
    
    def __init__(self, 
                 bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 n_population: int = 15,
                 max_iter_stage1: int = 100,
                 max_iter_stage2: int = 500,
                 tol: float = 1e-6,
                 seed: int = 42):
        self.bounds = bounds or PARAM_BOUNDS
        self.n_population = n_population
        self.max_iter_stage1 = max_iter_stage1
        self.max_iter_stage2 = max_iter_stage2
        self.tol = tol
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def calibrate(self,
                  field_data: pd.DataFrame,
                  oem_data: Optional[pd.DataFrame] = None,
                  initial_params: Optional[PhysicsParameters] = None) -> CalibrationResult:
        """
        Perform two-stage calibration.
        
        Args:
            field_data: DataFrame with mapped columns (time, soh_pct, temperature, SOC_pct, etc.)
            oem_data: Optional OEM baseline DataFrame
            initial_params: Optional initial parameter guess
            
        Returns:
            CalibrationResult with calibrated parameters and diagnostics
        """
        result = CalibrationResult(
            params=initial_params or PhysicsParameters(),
            param_bounds=self.bounds.copy(),
            param_status={},
        )
        
        # Prepare data
        data_prep = self._prepare_data(field_data, oem_data)
        if not data_prep["valid"]:
            result.warnings.append("Data preparation failed")
            return result
        
        # Store data for objective functions
        self.field_data = data_prep["field"]
        self.oem_data = data_prep["oem"]
        
        # Stage 1: Global optimization
        print("Starting Stage 1 calibration (global)...")
        stage1 = self._stage1_global_optimization(initial_params)
        result.stage1_result = stage1
        
        if not stage1["success"]:
            result.warnings.append("Stage 1 global optimization did not converge well")
            # Continue with best found
        
        # Update parameters with Stage 1 result
        params_stage1 = PhysicsParameters.from_array(stage1["x"])
        result.params = params_stage1
        
        # Stage 2: Local refinement
        print("Starting Stage 2 calibration (local)...")
        stage2 = self._stage2_local_refinement(params_stage1)
        result.stage2_result = stage2
        
        if not stage2["success"]:
            result.warnings.append("Stage 2 local refinement did not converge")
        
        # Final parameters
        result.params = PhysicsParameters.from_array(stage2["x"])
        result.success = stage1["success"] or stage2["success"]
        
        # Check parameter bounds
        result.param_status = self._check_parameter_bounds(result.params)
        
        # Generate model fit trajectory
        model = PhysicsDegradationModel(result.params)
        soh_pred, _, _ = model.predict_soh(
            time_hours=self.field_data['time_hours'].values,
            temperature=self.field_data['temperature'].values,
            soc=self.field_data['SOC_pct'].values,
            efc=self.field_data['EFC_per_day'].values,
            dod=self.field_data['Avg_DoD_pct'].values,
            c_rate=self.field_data['C_rate'].values,
            tmax=self.field_data['Tmax'].values,
            tmin=self.field_data['Tmin'].values,
            soh0=100.0
        )
        result.model_fit = soh_pred
        
        # Calculate final metrics
        result.metrics = self._calculate_final_metrics(result.params)
        
        return result
    
    def _prepare_data(self, field_df: pd.DataFrame, oem_df: Optional[pd.DataFrame]) -> Dict:
        """Prepare and validate data for calibration."""
        required_cols = ['time_parsed', 'soh_pct', 'temperature', 'SOC_pct', 'Avg_DoD_pct',
                        'EFC_per_day', 'Cycles_per_day', 'Charge_Duration_hr', 'Discharge_Duration_hr',
                        'C_rate']
        
        # Check required columns exist
        missing = [c for c in required_cols if c not in field_df.columns]
        if missing:
            # Try to derive from available columns
            field_df = self._derive_missing_columns(field_df)
            missing = [c for c in required_cols if c not in field_df.columns]
            if missing:
                return {"valid": False, "missing": missing}
        
        # Sort by time
        field_df = field_df.sort_values('time_parsed').reset_index(drop=True)
        
        # Handle missing values by interpolation
        field_df['time_parsed'] = pd.to_datetime(field_df['time_parsed'], errors='coerce')
        for col in required_cols:
            if col != 'time_parsed' and col in field_df.columns:
                field_df[col] = pd.to_numeric(field_df[col], errors='coerce')
                field_df[col] = field_df[col].interpolate(method='linear', limit_direction='both')
        
        # Remove rows with NaN in critical columns
        critical = ['time_parsed', 'soh_pct', 'temperature', 'SOC_pct']
        field_df = field_df.dropna(subset=[c for c in critical if c in field_df.columns])
        
        if len(field_df) < 5:
            return {"valid": False, "error": "Insufficient data points after cleaning"}
        
        # Convert time to hours from start
        t0 = field_df['time_parsed'].min()
        field_df['time_hours'] = (field_df['time_parsed'] - t0).dt.total_seconds() / 3600.0
        
        # Calculate Tmax and Tmin (rolling window if needed)
        if 'temperature' in field_df.columns:
            field_df['Tmax'] = field_df['temperature'].rolling(24, min_periods=1).max()
            field_df['Tmin'] = field_df['temperature'].rolling(24, min_periods=1).min()
        else:
            field_df['Tmax'] = field_df.get('temperature', 25.0)
            field_df['Tmin'] = field_df.get('temperature', 25.0)
        
        # Smooth SOH target
        field_df['soh_smooth'] = smooth_soh_target(
            field_df['soh_pct'].values, 
            field_df['time_hours'].values
        )
        
        # Detect step events
        step_indices = detect_soh_steps(field_df['soh_pct'].values, field_df['time_hours'].values)
        field_df['step_event'] = 0
        if len(step_indices) > 0:
            field_df.loc[step_indices, 'step_event'] = 1
        
        # Prepare OEM data
        oem_time, oem_soh = None, None
        if oem_df is not None:
            oem_time, oem_soh = prepare_oem_baseline(oem_df)
        
        return {
            "valid": True,
            "field": field_df,
            "oem": {"time": oem_time, "soh": oem_soh} if oem_time is not None else None,
        }
    
    def _derive_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive missing columns from available data."""
        df = df.copy()
        
        # Derive EFC_per_day from Cycles_per_day if available
        if 'EFC_per_day' not in df.columns and 'Cycles_per_day' in df.columns:
            df['EFC_per_day'] = df['Cycles_per_day']
        
        # Derive Cycles_per_day from EFC_per_day
        if 'Cycles_per_day' not in df.columns and 'EFC_per_day' in df.columns:
            df['Cycles_per_day'] = df['EFC_per_day']
        
        # Derive DoD if charge/discharge info available
        if 'Avg_DoD_pct' not in df.columns:
            if 'Charge_Duration_hr' in df.columns and 'Discharge_Duration_hr' in df.columns:
                # Simple approximation: total duration correlates with DoD
                total_dur = df['Charge_Duration_hr'] + df['Discharge_Duration_hr']
                df['Avg_DoD_pct'] = np.clip(total_dur * 10, 0, 100)
            elif 'C_rate' in df.columns:
                # Very rough estimate from C-rate
                df['Avg_DoD_pct'] = np.clip(df['C_rate'] * 50, 0, 100)
        
        # Default values for missing columns
        defaults = {
            'temperature': 25.0,
            'SOC_pct': 50.0,
            'Avg_DoD_pct': 50.0,
            'EFC_per_day': 1.0,
            'Cycles_per_day': 1.0,
            'Charge_Duration_hr': 2.0,
            'Discharge_Duration_hr': 2.0,
            'C_rate': 0.5,
        }
        
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
        
        return df
    
    def _stage1_global_optimization(self, 
                                     initial_params: Optional[PhysicsParameters]) -> Dict:
        """Stage 1: L-BFGS-B pre-fit optimization following V11 notebook methodology."""
        oem_time, oem_soh = None, None
        if self.oem_data is not None:
            if isinstance(self.oem_data, dict):
                oem_time = self.oem_data.get("time")
                oem_soh = self.oem_data.get("soh")
            elif isinstance(self.oem_data, (tuple, list)) and len(self.oem_data) == 2:
                oem_time, oem_soh = self.oem_data

        def objective(x):
            params = PhysicsParameters.from_array(x)
            model = PhysicsDegradationModel(params)
            
            # Predict SOH
            soh_pred, cal_loss, cyc_loss = model.predict_soh(
                time_hours=self.field_data['time_hours'].values,
                temperature=self.field_data['temperature'].values,
                soc=self.field_data['SOC_pct'].values,
                efc=self.field_data['EFC_per_day'].values,
                dod=self.field_data['Avg_DoD_pct'].values,
                c_rate=self.field_data['C_rate'].values,
                tmax=self.field_data['Tmax'].values,
                tmin=self.field_data['Tmin'].values,
                soh0=100.0
            )
            
            soh_target = self.field_data['soh_smooth'].values
            valid = ~np.isnan(soh_target) & ~np.isnan(soh_pred)
            
            if np.sum(valid) < 3:
                return 1e10
            
            rmse = np.sqrt(np.mean((soh_target[valid] - soh_pred[valid]) ** 2))
            
            # Add OEM baseline penalty if available
            if oem_time is not None and oem_soh is not None and len(oem_time) > 0:
                soh_at_oem = np.interp(oem_time, self.field_data['time_hours'].values, soh_pred)
                oem_rmse = np.sqrt(np.mean((oem_soh - soh_at_oem) ** 2))
                rmse += 0.5 * oem_rmse
            
            return rmse
        
        bounds_list = [self.bounds.get(k, (0.01, 10.0)) for k in [
            'A_cal', 'A_cyc', 'Ea_cal', 'Ea_cyc', 'alpha', 'beta',
            'k_SOC_high', 'k_SOC_reward', 'k_SOC_low', 'gamma_DoD',
            'k_C', 'k_Tmax', 'k_Tmin'
        ]]
        
        x0 = initial_params.to_array() if initial_params else PhysicsParameters().to_array()
        
        try:
            result = minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds_list,
                options={"maxiter": min(self.max_iter_stage1, 200), "ftol": 1e-8},
            )
            
            return {
                "success": result.success,
                "x": result.x,
                "fun": result.fun,
                "message": result.message,
                "nfev": result.nfev,
                "nit": result.nit,
            }
        except Exception as e:
            return {
                "success": False,
                "x": x0,
                "fun": np.inf,
                "message": str(e),
                "nfev": 0,
                "nit": 0,
            }
    
    def _stage2_local_refinement(self, params: PhysicsParameters) -> Dict:
        """Stage 2: Least squares local refinement."""
        oem_time, oem_soh = None, None
        if self.oem_data is not None:
            if isinstance(self.oem_data, dict):
                oem_time = self.oem_data.get("time")
                oem_soh = self.oem_data.get("soh")
            elif isinstance(self.oem_data, (tuple, list)) and len(self.oem_data) == 2:
                oem_time, oem_soh = self.oem_data
        
        def residuals(x):
            params = PhysicsParameters.from_array(x)
            model = PhysicsDegradationModel(params)
            
            soh_pred, _, _ = model.predict_soh(
                time_hours=self.field_data['time_hours'].values,
                temperature=self.field_data['temperature'].values,
                soc=self.field_data['SOC_pct'].values,
                efc=self.field_data['EFC_per_day'].values,
                dod=self.field_data['Avg_DoD_pct'].values,
                c_rate=self.field_data['C_rate'].values,
                tmax=self.field_data['Tmax'].values,
                tmin=self.field_data['Tmin'].values,
                soh0=100.0
            )
            
            soh_target = self.field_data['soh_smooth'].values
            valid = ~np.isnan(soh_target) & ~np.isnan(soh_pred)
            
            if np.sum(valid) < 3:
                return np.array([1e10])
            
            res = soh_pred[valid] - soh_target[valid]
            
            # Add OEM residuals
            if oem_time is not None and oem_soh is not None and len(oem_time) > 0:
                soh_at_oem = np.interp(oem_time, self.field_data['time_hours'].values, soh_pred)
                oem_res = soh_at_oem - oem_soh
                res = np.concatenate([res, 0.5 * oem_res])
            
            return res
        
        bounds_list = [self.bounds.get(k, (0.01, 10.0)) for k in [
            'A_cal', 'A_cyc', 'Ea_cal', 'Ea_cyc', 'alpha', 'beta',
            'k_SOC_high', 'k_SOC_reward', 'k_SOC_low', 'gamma_DoD',
            'k_C', 'k_Tmax', 'k_Tmin'
        ]]
        
        try:
            result = least_squares(
                residuals,
                params.to_array(),
                bounds=([b[0] for b in bounds_list], [b[1] for b in bounds_list]),
                max_nfev=min(self.max_iter_stage2, 300),
                ftol=1e-6,
                xtol=1e-6,
                verbose=0,
            )
            
            return {
                "success": result.success,
                "x": result.x,
                "fun": result.fun,
                "cost": result.cost,
                "message": result.message,
                "nfev": result.nfev,
            }
        except Exception as e:
            return {
                "success": False,
                "x": params.to_array(),
                "fun": np.array([np.inf]),
                "cost": np.inf,
                "message": str(e),
                "nfev": 0,
            }
    
    def _check_parameter_bounds(self, params: PhysicsParameters) -> Dict[str, str]:
        """Check if parameters are at bounds (warnings)."""
        status = {}
        param_dict = params.to_dict()
        
        for name, value in param_dict.items():
            if name in self.bounds:
                low, high = self.bounds[name]
                if abs(value - low) < 1e-6:
                    status[name] = "AT_LOWER_BOUND"
                elif abs(value - high) < 1e-6:
                    status[name] = "AT_UPPER_BOUND"
                else:
                    status[name] = "OK"
            else:
                status[name] = "UNBOUNDED"
        
        return status
    
    def _calculate_final_metrics(self, params: PhysicsParameters) -> Dict[str, float]:
        """Calculate final model performance metrics."""
        model = PhysicsDegradationModel(params)
        
        soh_pred, cal_loss, cyc_loss = model.predict_soh(
            time_hours=self.field_data['time_hours'].values,
            temperature=self.field_data['temperature'].values,
            soc=self.field_data['SOC_pct'].values,
            efc=self.field_data['EFC_per_day'].values,
            dod=self.field_data['Avg_DoD_pct'].values,
            c_rate=self.field_data['C_rate'].values,
            tmax=self.field_data['Tmax'].values,
            tmin=self.field_data['Tmin'].values,
            soh0=100.0
        )
        
        soh_obs = self.field_data['soh_pct'].values
        soh_smooth = self.field_data['soh_smooth'].values
        
        metrics_raw = calculate_metrics(soh_obs, soh_pred)
        metrics_smooth = calculate_metrics(soh_smooth, soh_pred)
        
        # OEM baseline metrics
        oem_rmse = np.nan
        if self.oem_data is not None:
            if isinstance(self.oem_data, dict):
                oem_time = self.oem_data.get("time")
                oem_soh = self.oem_data.get("soh")
            elif isinstance(self.oem_data, (tuple, list)) and len(self.oem_data) == 2:
                oem_time, oem_soh = self.oem_data
            else:
                oem_time, oem_soh = None, None
            if oem_time is not None and oem_soh is not None and len(oem_time) > 0:
                soh_at_oem = np.interp(oem_time, self.field_data['time_hours'].values, soh_pred)
                oem_rmse = float(np.sqrt(np.mean((oem_soh - soh_at_oem) ** 2)))
        
        return {
            "rmse_vs_raw": metrics_raw["rmse"],
            "mae_vs_raw": metrics_raw["mae"],
            "r2_vs_raw": metrics_raw["r2"],
            "rmse_vs_smooth": metrics_smooth["rmse"],
            "mae_vs_smooth": metrics_smooth["mae"],
            "r2_vs_smooth": metrics_smooth["r2"],
            "rmse_vs_oem": oem_rmse if not np.isnan(oem_rmse) else None,
            "total_calendar_loss": float(cal_loss[-1]),
            "total_cycle_loss": float(cyc_loss[-1]),
            "total_modeled_loss": float(cal_loss[-1] + cyc_loss[-1]),
            "observed_loss": float(100 - soh_obs[-1]),
        }


def run_calibration(field_data: pd.DataFrame,
                    oem_data: Optional[pd.DataFrame] = None,
                    initial_params: Optional[PhysicsParameters] = None) -> CalibrationResult:
    """Convenience function to run calibration."""
    calibrator = TwoStageCalibrator()
    return calibrator.calibrate(field_data, oem_data, initial_params)