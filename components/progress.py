import streamlit as st
from typing import Optional, Dict, Any
from utils.session_state import SessionStateManager
from utils.formatting import progress_bar, format_percentage, format_years


def render_calibration_progress(calibration_result: Optional[Any] = None) -> None:
    """Render calibration progress indicator."""
    
    if calibration_result is None:
        # Initial state
        st.markdown("### ⚙️ Calibration")
        st.info("Calibration has not been run yet. Click 'Run Calibration' to start.")
        
        st.markdown("#### Calibration Steps")
        steps = [
            "1. Data Preparation",
            "2. SOH Step Event Detection", 
            "3. Smooth Target Generation",
            "4. OEM Baseline Preparation",
            "5. Stage-1 Global Calibration",
            "6. Stage-2 Local Refinement",
            "7. Model Validation",
        ]
        
        for step in steps:
            st.markdown(f"○ {step}")
        
        return
    
    # Running state
    st.markdown("### ⚙️ Calibration Progress")
    
    if not calibration_result.success:
        st.warning("⚠ Calibration completed with warnings or errors")
    
    # Show calibration stages
    st.markdown("#### Calibration Stages")
    
    stages = [
        ("Data Preparation", True),
        ("SOH Step Event Detection", True),
        ("Smooth Target Generation", True),
        ("OEM Baseline Preparation", True),
        ("Stage-1 Global Calibration", calibration_result.stage1_result is not None),
        ("Stage-2 Local Refinement", calibration_result.stage2_result is not None),
        ("Model Validation", calibration_result.success),
    ]
    
    for name, completed in stages:
        if completed:
            st.markdown(f"✅ {name}")
        else:
            st.markdown(f"⏳ {name}")
    
    # Show stage results
    if calibration_result.stage1_result:
        st.markdown("#### Stage-1 Results")
        s1 = calibration_result.stage1_result
        st.metric("Objective (RMSE)", f"{s1.get('fun', 0):.6f}")
        st.metric("Iterations", s1.get('nit', 0))
    
    if calibration_result.stage2_result:
        st.markdown("#### Stage-2 Results")
        s2 = calibration_result.stage2_result
        st.metric("Final Objective", f"{s2.get('fun', [0])[0] if isinstance(s2.get('fun'), list) else s2.get('fun', 0):.6f}")
        st.metric("Iterations", s2.get('nfev', 0))
    
    # Parameter status
    if calibration_result.param_status:
        st.markdown("#### Parameter Status")
        
        for name, status in calibration_result.param_status.items():
            if status == "AT_LOWER_BOUND":
                st.warning(f"⚠ {name}: AT LOWER BOUND - may need more data")
            elif status == "AT_UPPER_BOUND":
                st.warning(f"⚠ {name}: AT UPPER BOUND - may need more data")
            else:
                st.success(f"✓ {name}: {status}")
        
        # Show warnings about bound issues
        bound_warnings = [name for name, s in calibration_result.param_status.items() 
                          if s != "OK"]
        if bound_warnings:
            st.markdown("### ⚠ Parameter Bound Warnings")
            st.info("Some parameters reached their bounds. This may indicate:")
            st.markdown("""
            - Insufficient data for certain parameters
            - Parameters that cannot be uniquely identified from available data
            - The model may need regularization or additional constraints
            
            These parameters should be interpreted with caution.
            """)
    
    # Metrics
    if calibration_result.metrics:
        st.markdown("#### Calibration Metrics")
        m = calibration_result.metrics
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE vs Raw", f"{m.get('rmse_vs_raw', 0):.4f}")
        with col2:
            st.metric("RMSE vs Smooth", f"{m.get('rmse_vs_smooth', 0):.4f}")
        with col3:
            st.metric("R² vs Raw", f"{m.get('r2_vs_raw', 0):.4f}")
        with col4:
            st.metric("R² vs Smooth", f"{m.get('r2_vs_smooth', 0):.4f}")
        
        if m.get('rmse_vs_oem'):
            st.metric("RMSE vs OEM", f"{m['rmse_vs_oem']:.4f}")
    
    # Diagnostics
    st.markdown("#### Calibration Diagnostics")
    
    if hasattr(calibration_result, 'convergence_info'):
        ci = calibration_result.convergence_info
        if ci:
            st.json(ci)
    
    if calibration_result.warnings:
        st.markdown("#### Warnings")
        for w in calibration_result.warnings:
            st.warning(f"⚠ {w}")
    
    if calibration_result.errors:
        st.markdown("#### Errors")
        for e in calibration_result.errors:
            st.error(f"✕ {e}")


def render_forecast_progress(forecast_result: Optional[Any] = None) -> None:
    """Render forecast progress indicator."""
    
    if forecast_result is None:
        st.markdown("### 🔮 Forecast")
        st.info("Run calibration first, then click 'Run Forecast'.")
        return
    
    st.markdown("### 🔮 Forecast Results")
    
    # EOL prediction
    eol = forecast_result.eol_info
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if eol['eol_reached']:
            st.metric("Predicted EOL", f"{eol.get('years_to_eol', 0):.1f} years")
        else:
            st.metric("Predicted EOL", f"> {eol.get('years_to_eol', 0):.1f} years")
    
    with col2:
        st.metric("Final SOH", f"{forecast_result.forecast_soh[-1]:.1f}%")
    
    with col3:
        st.metric("Calendar Loss", f"{forecast_result.calendar_loss_forecast[-1]:.2f}%")
    
    with col4:
        st.metric("Cycle Loss", f"{forecast_result.cycle_loss_forecast[-1]:.2f}%")
    
    # Forecast details
    st.markdown("#### Forecast Details")
    
    if eol['eol_reached']:
        st.success(f"EOL predicted at {eol.get('years_to_eol', 0):.1f} years")
    else:
        st.info(f"EOL not reached within forecast horizon ({eol.get('years_to_eol', 0):.1f} years)")
    
    # Warnings
    if forecast_result.warnings:
        st.markdown("#### Forecast Warnings")
        for w in forecast_result.warnings:
            st.warning(f"⚠ {w}")
    
    # Performance metrics
    if forecast_result.metrics:
        st.markdown("#### Model Performance")
        metrics = forecast_result.metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        with col2:
            st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
        with col3:
            st.metric("R²", f"{metrics.get('r2', 0):.4f}")
        with col4:
            mape = metrics.get('mape', 0)
            st.metric("MAPE", f"{mape:.2f}%" if mape == mape else "N/A")


def render_scenario_progress(scenario_results: Optional[Dict] = None) -> None:
    """Render scenario analysis progress."""
    
    if scenario_results is None or not scenario_results:
        st.markdown("### 📈 Scenario Analysis")
        st.info("Run calibration and forecasting first, then generate scenarios.")
        return
    
    st.markdown("### 📈 Scenario Analysis Results")
    
    # Show each scenario
    for name, result in scenario_results.items():
        if isinstance(result, dict) and 'scenario_name' in result:
            # Render scenario card
            st.markdown(f"#### {result['scenario_name'].upper()} CASE")
            st.markdown(f"*{result.get('description', '')}*")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("EOL", format_years(result.get('EOL_year', 0)))
            with col2:
                st.metric("Final SOH", f"{result.get('Final_SOH_pct', 0):.1f}%")
            with col3:
                st.metric("EFC/day", f"{result.get('EFC_per_day', 0):.2f}")
            with col4:
                st.metric("Avg DoD", f"{result.get('Avg_DoD_pct', 0):.1f}%")


def render_overall_progress() -> None:
    """Render the overall workflow progress."""
    st.markdown("### 📋 Workflow Progress")
    
    steps = [
        ("Upload", 0),
        ("Validation", 1),
        ("Exploration", 2),
        ("OEM Baseline", 3),
        ("Model Selection", 4),
        ("Calibration", 5),
        ("Forecast", 6),
        ("Scenarios", 7),
        ("Comparison", 8),
        ("Report", 9),
    ]
    
    completed = st.session_state.get("step_completed", {})
    
    for name, step in steps:
        is_completed = completed.get(step, False)
        is_current = step == st.session_state.get("current_step", 0)
        
        if is_completed:
            st.markdown(f"✅ **{name}**")
        elif is_current:
            st.markdown(f"▶ **{name}** *(current)*")
        else:
            st.markdown(f"○ **{name}**")
        
        # Show status indicators
        if is_completed and not is_current:
            st.markdown("↳ Completed")