import streamlit as st
from typing import Any, Dict, List, Optional
import pandas as pd


class SessionStateManager:
    """Manages Streamlit session state for the BESS application."""
    
    # Define all session state keys
    KEYS = {
        # Data upload
        "uploaded_files": [],
        "file_metadata": {},
        "selected_sheets": {},
        
        # Column mapping
        "column_mappings": {},
        "mapped_data": {},
        "clean_data": None,
        
        # Validation
        "validation_results": None,
        "validation_passed": False,
        
        # OEM baseline
        "oem_data": None,
        "oem_metadata": {},
        "oem_validation": None,
        
        # Model selection
        "selected_model": "Physics-Based",
        
        # Calibration
        "calibration_result": None,
        "calibrated_params": None,
        "calibration_warnings": [],
        "calibration_metrics": {},
        
        # Forecasting
        "forecast_result": None,
        "forecast_scenario": "middle",
        "forecast_years": 10.0,
        
        # Scenarios
        "scenario_results": {},
        
        # Model comparison
        "pinn_model": None,
        "pinn_trained": False,
        "pinn_forecast": None,
        "comparison_results": None,
        
        # Uncertainty
        "uncertainty_result": None,
        
        # Diagnostics
        "diagnostics_result": None,
        
        # Report
        "report_data": None,
        
        # Workflow state
        "current_step": 0,
        "step_completed": {
            0: False,  # Upload
            1: False,  # Validation
            2: False,  # Exploration
            3: False,  # OEM
            4: False,  # Model Selection
            5: False,  # Calibration
            6: False,  # Forecast
            7: False,  # Scenarios
            8: False,  # Comparison
            9: False,  # Report
        },
        
        # Settings
        "eol_threshold": 65.0,
        
        # UI state
        "show_debug": False,
        "last_error": None,
    }
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize all session state keys with defaults."""
        for key, default in cls.KEYS.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a session state value with optional default."""
        return st.session_state.get(key, default)
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a session state value."""
        st.session_state[key] = value
    
    @classmethod
    def update(cls, updates: Dict[str, Any]) -> None:
        """Update multiple session state values."""
        for key, value in updates.items():
            st.session_state[key] = value
    
    @classmethod
    def mark_step_completed(cls, step: int) -> None:
        """Mark a workflow step as completed."""
        completed = st.session_state.get("step_completed", {})
        completed[step] = True
        st.session_state["step_completed"] = completed
        st.session_state["current_step"] = max(st.session_state.get("current_step", 0), step + 1)
    
    @classmethod
    def is_step_completed(cls, step: int) -> bool:
        """Check if a workflow step is completed."""
        completed = st.session_state.get("step_completed", {})
        return completed.get(step, False)
    
    @classmethod
    def can_access_step(cls, step: int) -> bool:
        """Check if user can access a step (all previous steps completed)."""
        for i in range(step):
            if not cls.is_step_completed(i):
                return False
        return True
    
    @classmethod
    def reset_workflow(cls) -> None:
        """Reset workflow state (keep uploaded data)."""
        steps_to_reset = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        completed = st.session_state.get("step_completed", {})
        for step in steps_to_reset:
            completed[step] = False
        st.session_state["step_completed"] = completed
        st.session_state["current_step"] = 0
        
        # Clear derived results
        for key in ["validation_results", "validation_passed", "calibration_result", 
                    "calibrated_params", "forecast_result", "scenario_results",
                    "pinn_model", "pinn_forecast", "comparison_results",
                    "uncertainty_result", "diagnostics_result", "report_data"]:
            st.session_state[key] = cls.KEYS.get(key)
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all session state."""
        for key, default in cls.KEYS.items():
            st.session_state[key] = default
    
    @classmethod
    def get_uploaded_files_summary(cls) -> List[Dict[str, Any]]:
        """Get summary of uploaded files with full metadata for rendering."""
        files = cls.get("uploaded_files") or []
        metadata = cls.get("file_metadata") or {}
        
        summary = []
        for f in files:
            meta = metadata.get(f.name, {})
            summary.append({
                "name": f.name,
                "file_type": meta.get("file_type", "unknown"),
                "rows": meta.get("rows", 0),
                "columns": meta.get("columns", 0),
                "date_range": meta.get("date_range", (None, None)),
                "missing_values": meta.get("missing_values", 0),
                "duplicate_rows": meta.get("duplicate_rows", 0),
                "detected_soh_column": meta.get("detected_soh_column", ""),
                "detected_temperature_columns": meta.get("detected_temperature_columns", []),
                "detected_soc_column": meta.get("detected_soc_column", ""),
                "detected_dod_column": meta.get("detected_dod_column", ""),
                "detected_efc_column": meta.get("detected_efc_column", ""),
                "validation_status": meta.get("validation_status", "pending"),
                "size_mb": f.size / (1024 * 1024) if hasattr(f, 'size') else 0,
            })
        return summary
    
    @classmethod
    def get_clean_data(cls) -> Optional[pd.DataFrame]:
        """Get the cleaned/merged data from all uploaded files."""
        return cls.get("clean_data")
    
    @classmethod
    def set_clean_data(cls, df: pd.DataFrame) -> None:
        """Set the cleaned/merged data."""
        cls.set("clean_data", df)
    
    @classmethod
    def add_file(cls, file_obj, metadata: Dict[str, Any]) -> None:
        """Add an uploaded file."""
        files = cls.get("uploaded_files") or []
        metadata_dict = cls.get("file_metadata") or {}
        
        # Check if already exists
        if not any(f.name == file_obj.name for f in files):
            files.append(file_obj)
        
        metadata_dict[file_obj.name] = metadata
        
        cls.set("uploaded_files", files)
        cls.set("file_metadata", metadata_dict)
    
    @classmethod
    def remove_file(cls, file_name: str) -> None:
        """Remove an uploaded file."""
        files = cls.get("uploaded_files") or []
        metadata = cls.get("file_metadata") or {}
        
        files = [f for f in files if f.name != file_name]
        if file_name in metadata:
            del metadata[file_name]
        
        cls.set("uploaded_files", files)
        cls.set("file_metadata", metadata)
        
        # Reset dependent steps
        cls.mark_step_completed(0)
        for step in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            completed = cls.get("step_completed")
            completed[step] = False
            cls.set("step_completed", completed)


# Convenience functions
def init_session_state():
    """Initialize session state - call at app startup."""
    SessionStateManager.initialize()


def get_state(key: str, default: Any = None) -> Any:
    """Get session state value with optional default."""
    return SessionStateManager.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Set session state value."""
    SessionStateManager.set(key, value)