import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

from backend.physics_model import PhysicsDegradationModel, PhysicsParameters
from backend.forecasting import ForecastResult


@dataclass
class PINNConfig:
    """Configuration for PINN model."""
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 32])
    activation: str = "tanh"
    
    # Physics loss weights
    data_loss_weight: float = 1.0
    physics_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0
    
    # Training
    learning_rate: float = 1e-3
    epochs: int = 1000
    batch_size: int = 32
    patience: int = 50
    
    # Physics constraints
    enforce_monotonic: bool = True  # SOH should not increase
    enforce_eol_threshold: float = 65.0  # SOH at EOL
    
    # Regularization
    l2_reg: float = 1e-4
    dropout: float = 0.0


class PINNModel:
    """
    Physics-Informed Neural Network for BESS degradation.
    
    This is a modular interface that can be implemented with different
    backend frameworks (PyTorch, TensorFlow, JAX).
    
    The PINN learns the degradation function while respecting physical constraints:
    - Data loss: Match observed SOH
    - Physics loss: Satisfy degradation ODE
    - Boundary loss: SOH(0) = 100%, SOH(EOL) = 65%
    """
    
    def __init__(self, config: Optional[PINNConfig] = None):
        self.config = config or PINNConfig()
        self.model = None
        self.is_trained = False
        self.training_history = {"loss": [], "data_loss": [], "physics_loss": [], "boundary_loss": []}
        self.scaler_params: Dict[str, Tuple[float, float]] = {}
    
    def prepare_data(self, field_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from field data."""
        df = field_data.sort_values('time_parsed').reset_index(drop=True)
        
        # Time in years
        t0 = df['time_parsed'].min()
        time_years = (df['time_parsed'] - t0).dt.total_seconds() / (365.25 * 24 * 3600)
        
        # Features: time, temperature, SOC, DoD, EFC, C-rate, Tmax, Tmin
        features = np.column_stack([
            time_years.values,
            pd.to_numeric(df['temperature'], errors='coerce').values,
            pd.to_numeric(df['SOC_pct'], errors='coerce').values / 100.0,
            pd.to_numeric(df['Avg_DoD_pct'], errors='coerce').values / 100.0,
            pd.to_numeric(df['EFC_per_day'], errors='coerce').values,
            pd.to_numeric(df['C_rate'], errors='coerce').values,
            pd.to_numeric(df['Tmax'], errors='coerce').values,
            pd.to_numeric(df['Tmin'], errors='coerce').values,
        ])
        
        # Target: SOH
        target = pd.to_numeric(df['soh_pct'], errors='coerce').values / 100.0
        
        # Remove NaN rows
        valid = ~np.isnan(features).any(axis=1) & ~np.isnan(target)
        features = features[valid]
        target = target[valid]
        time_years = time_years.values[valid]
        
        return features, target, time_years
    
    def fit(self, field_data: pd.DataFrame, physics_model: Optional[PhysicsDegradationModel] = None) -> Dict[str, Any]:
        """
        Train the Physics-Informed ML Residual Model (Cell 26 from v11 notebook).
        Fits a HistGradientBoostingRegressor on the systematic residual between
        physics degradation prediction and actual field observations.
        """
        self.field_data = field_data.copy()
        self.physics_model = physics_model
        
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            df = field_data.sort_values('time_parsed' if 'time_parsed' in field_data.columns else field_data.columns[0]).reset_index(drop=True)
            n = len(df)
            if n < 10:
                return {"success": False, "message": "Insufficient data points for ML training"}
            
            # Extract time_hours
            if 'time_hours' in df.columns:
                time_hours = pd.to_numeric(df['time_hours'], errors='coerce').fillna(0.0).values
            elif 'time_parsed' in df.columns:
                t_dt = pd.to_datetime(df['time_parsed'], errors='coerce')
                t0 = t_dt.iloc[0] if len(t_dt) > 0 else pd.Timestamp('2024-01-01')
                time_hours = (t_dt - t0).dt.total_seconds().values / 3600.0
            else:
                time_hours = np.arange(n, dtype=float) * 24.0

            def _get_col_arr(col_name, default_val):
                if col_name in df.columns:
                    s = pd.to_numeric(df[col_name], errors='coerce')
                    if isinstance(default_val, np.ndarray):
                        return np.where(s.notna().values, s.values, default_val)
                    return s.fillna(default_val).values
                if isinstance(default_val, np.ndarray):
                    return default_val.copy()
                return np.full(n, default_val, dtype=float)

            temp = _get_col_arr('temperature', 25.0)
            soc = _get_col_arr('SOC_pct', 50.0)
            efc = _get_col_arr('EFC_per_day', 1.0)
            dod = _get_col_arr('Avg_DoD_pct', 95.0) / 100.0
            crate = _get_col_arr('C_rate', 0.25)
            tmax = _get_col_arr('Tmax', temp)
            tmin = _get_col_arr('Tmin', temp)

            # 1. Physics baseline prediction
            if physics_model is not None:
                soh_physics, dcal, dcyc = physics_model.predict_soh(
                    time_hours=time_hours,
                    temperature=temp,
                    soc=soc,
                    efc=efc,
                    dod=dod,
                    c_rate=crate,
                    tmax=tmax,
                    tmin=tmin,
                    soh0=100.0
                )
            else:
                soh_physics = np.full(n, 100.0)
                dcal = np.zeros(n)
                dcyc = np.zeros(n)
            
            obs_soh = pd.to_numeric(df['soh_pct'], errors='coerce').values
            residual = obs_soh - soh_physics
            
            # 2. Build feature matrix matching v11 notebook cell 26
            
            X = pd.DataFrame({
                "Day": np.arange(n, dtype=float),
                "Cumulative_EFC": np.cumsum(efc),
                "EFC_delta_vs_OEM": efc - 1.0,
                "DOD_delta_vs_OEM": dod - 1.0,
                "Tavg_delta": temp - 35.0,
                "Tmax_delta": tmax - 40.0,
                "SOC_high_stress": np.maximum(0.0, soc - 80.0),
                "SOC_mid_reward_distance": np.abs(soc - 50.0),
                "SOC_low_stress": np.maximum(0.0, 20.0 - soc),
                "Physics_dcal_pp": dcal,
                "Physics_dcyc_pp": dcyc,
            })
            
            split = int(0.70 * n)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = residual[:split], residual[split:]
            
            ml_model = HistGradientBoostingRegressor(
                max_iter=150, 
                max_depth=3, 
                learning_rate=0.03, 
                random_state=42
            )
            ml_model.fit(X_train, y_train)
            
            train_pred = ml_model.predict(X_train)
            test_pred = ml_model.predict(X_test) if len(X_test) > 0 else train_pred
            zero_test = np.zeros_like(y_test)
            
            train_mae = float(mean_absolute_error(y_train, train_pred))
            test_mae = float(mean_absolute_error(y_test, test_pred)) if len(X_test) > 0 else train_mae
            zero_mae = float(mean_absolute_error(y_test, zero_test)) if len(X_test) > 0 else 1.0
            test_rmse = float(mean_squared_error(y_test, test_pred) ** 0.5) if len(X_test) > 0 else train_mae
            
            # Check if ML beats zero-residual baseline
            usable = test_mae < zero_mae
            
            self.model = ml_model
            self.feature_columns = list(X.columns)
            self.is_trained = True
            self.metrics = {
                "train_mae": train_mae,
                "test_mae": test_mae,
                "zero_mae": zero_mae,
                "test_rmse": test_rmse,
                "usable": usable,
                "r2": float(r2_score(obs_soh, soh_physics + ml_model.predict(X))),
            }
            
            return {
                "success": True,
                "usable": usable,
                "metrics": self.metrics,
                "message": "Guarded Physics-Informed ML Residual Model trained successfully!"
            }
            
        except Exception as e:
            return {"success": False, "message": f"Training failed: {str(e)}"}
    
    def forecast(self,
                 field_data: pd.DataFrame,
                 forecast_years: float = 10.0,
                 scenario: str = "middle") -> ForecastResult:
        """Generate forecast combining Physics Model with Guarded ML Residuals."""
        from backend.forecasting import PhysicsForecaster
        from backend.physics_model import predict_eol
        from utils.constants import EOL_SOH_THRESHOLD
        
        # 1. Physics forecast
        if self.physics_model is not None:
            forecaster = PhysicsForecaster(self.physics_model, field_data)
            base_fc = forecaster.forecast(forecast_years, scenario)
        else:
            raise ValueError("Physics model is required for PINN/Guarded ML forecast.")
        
        if not self.is_trained or self.model is None:
            return base_fc
        
        # 2. Build future features
        n_days = len(base_fc.forecast_dates)
        day_offset = len(field_data)
        cum_efc_start = float(field_data.get('EFC_per_day', pd.Series([1.0])).sum())
        
        efc_val = float(field_data.get('EFC_per_day', pd.Series([1.0])).median())
        dod_val = float(field_data.get('Avg_DoD_pct', pd.Series([95.0])).median()) / 100.0
        soc_val = float(field_data.get('SOC_pct', pd.Series([50.0])).median())
        temp_val = float(field_data.get('temperature', pd.Series([25.0])).median())
        tmax_val = float(field_data.get('Tmax', pd.Series([30.0])).median())
        
        X_fc = pd.DataFrame({
            "Day": np.arange(n_days, dtype=float) + float(day_offset),
            "Cumulative_EFC": cum_efc_start + np.cumsum(np.full(n_days, efc_val)),
            "EFC_delta_vs_OEM": np.full(n_days, efc_val - 1.0),
            "DOD_delta_vs_OEM": np.full(n_days, dod_val - 1.0),
            "Tavg_delta": np.full(n_days, temp_val - 35.0),
            "Tmax_delta": np.full(n_days, tmax_val - 40.0),
            "SOC_high_stress": np.full(n_days, max(0.0, soc_val - 80.0)),
            "SOC_mid_reward_distance": np.full(n_days, abs(soc_val - 50.0)),
            "SOC_low_stress": np.full(n_days, max(0.0, 20.0 - soc_val)),
            "Physics_dcal_pp": base_fc.calendar_loss_forecast,
            "Physics_dcyc_pp": base_fc.cycle_loss_forecast,
        })
        
        # Align features
        X_fc = X_fc[self.feature_columns]
        ml_corr = self.model.predict(X_fc)
        # Clip to +/- 1.5% and decay exponentially with 5-year half life
        ml_corr = np.clip(ml_corr, -1.5, 1.5)
        years = np.arange(n_days) / 365.25
        decay = np.exp(-years / 5.0)
        ml_corr = ml_corr * decay
        
        corrected_soh = base_fc.forecast_soh + ml_corr
        corrected_soh = np.clip(corrected_soh, 0.0, 100.0)
        corrected_soh = np.minimum.accumulate(corrected_soh)
        
        forecast_time_hours = np.arange(24, n_days * 24 + 24, 24.0)
        eol_info = predict_eol(corrected_soh, forecast_time_hours, EOL_SOH_THRESHOLD)
        
        return ForecastResult(
            historical_dates=base_fc.historical_dates,
            historical_soh=base_fc.historical_soh,
            historical_fit=base_fc.historical_fit,
            forecast_dates=base_fc.forecast_dates,
            forecast_soh=corrected_soh,
            calendar_loss_historical=base_fc.calendar_loss_historical,
            cycle_loss_historical=base_fc.cycle_loss_historical,
            calendar_loss_forecast=base_fc.calendar_loss_forecast,
            cycle_loss_forecast=base_fc.cycle_loss_forecast,
            eol_info=eol_info,
            metrics=getattr(self, 'metrics', base_fc.metrics),
            parameters=base_fc.parameters,
        )


def create_pinn_model(config: Optional[PINNConfig] = None) -> PINNModel:
    """Factory function to create PINN model."""
    return PINNModel(config)