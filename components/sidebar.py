import streamlit as st
from typing import List, Dict, Any, Optional
from utils.session_state import SessionStateManager


def render_sidebar() -> None:
    """Render the main sidebar with workflow navigation."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 16px 0; border-bottom: 1px solid #e1e8ed;">
            <h2 style="margin: 0; color: #2c3e50;">BESS Analytics</h2>
            <p style="margin: 4px 0 0; color: #7f8c8d; font-size: 0.85rem;">SOH Degradation & Forecasting</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Workflow steps
        steps = [
            (0, "📁", "Data Upload"),
            (1, "✅", "Validation"),
            (2, "📊", "Exploration"),
            (3, "🏭", "OEM Baseline"),
            (4, "🤖", "Model Selection"),
            (5, "⚙️", "Calibration"),
            (6, "🔮", "Forecast"),
            (7, "📈", "Scenarios"),
            (8, "⚖️", "Comparison"),
            (9, "📋", "Report"),
        ]
        
        current_step = st.session_state.get("current_step", 0)
        completed = st.session_state.get("step_completed", {})
        
        st.markdown("### Workflow Progress")
        
        for step_num, icon, label in steps:
            is_completed = completed.get(step_num, False)
            is_current = step_num == current_step
            is_accessible = step_num <= current_step or (step_num == current_step + 1 and is_completed)
            
            if is_completed:
                status = "✓"
                style = "color: #27ae60;"
            elif is_current:
                status = "▶"
                style = "color: #3498db; font-weight: 600;"
            elif is_accessible:
                status = "○"
                style = "color: #f39c12;"
            else:
                status = "○"
                style = "color: #bdc3c7;"
            
            col1, col2 = st.columns([0.15, 0.85])
            with col1:
                st.markdown(f'<span style="{style} font-size: 1.2rem;">{status}</span>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<span style="{style} font-size: 0.9rem;">{label}</span>', unsafe_allow_html=True)
        
        st.divider()
        
        # Quick stats
        if SessionStateManager.is_step_completed(0):
            st.markdown("### Quick Stats")
            files = SessionStateManager.get_uploaded_files_summary()
            st.metric("Files Uploaded", len(files))
            
            if SessionStateManager.get("clean_data") is not None:
                df = SessionStateManager.get("clean_data")
                st.metric("Total Rows", f"{len(df):,}")
                st.metric("Date Span", f"{(df['time_parsed'].max() - df['time_parsed'].min()).days} days")
        
        if SessionStateManager.is_step_completed(5):
            st.markdown("### Calibration")
            params = SessionStateManager.get("calibrated_params")
            if params:
                st.metric("Parameters", len(params.to_dict()))
                if SessionStateManager.get("calibration_metrics"):
                    rmse = SessionStateManager.get("calibration_metrics").get("rmse_vs_smooth", 0)
                    st.metric("RMSE (smooth)", f"{rmse:.3f}")
        
        if SessionStateManager.is_step_completed(6):
            st.markdown("### Forecast")
            forecast = SessionStateManager.get("forecast_result")
            if forecast:
                st.metric("Forecast Years", f"{st.session_state.get('forecast_years', 10):.0f}")
                eol = forecast.eol_info.get("years_to_eol", 0)
                st.metric("EOL", format_years(eol))
        
        st.divider()
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.session_state["show_debug"] = st.checkbox("Show Debug Info", 
                value=st.session_state.get("show_debug", False))
            
            if st.button("🔄 Reset Workflow", use_container_width=True):
                SessionStateManager.reset_workflow()
                st.rerun()
            
            if st.button("🗑️ Clear All Data", use_container_width=True):
                SessionStateManager.reset_all()
                st.rerun()
        
        # Version info
        st.markdown("""
        <div style="position: fixed; bottom: 0; left: 0; right: 0; 
                    padding: 12px; text-align: center; color: #95a5a6; 
                    font-size: 0.75rem; background: white; border-top: 1px solid #e1e8ed;">
            BESS Analytics v1.0<br>
            Built with Streamlit
        </div>
        """, unsafe_allow_html=True)


def render_step_indicator(current: int, total: int = 10) -> None:
    """Render a horizontal step indicator."""
    steps = [
        (0, "Upload"), (1, "Validate"), (2, "Explore"), (3, "OEM"),
        (4, "Model"), (5, "Calibrate"), (6, "Forecast"), (7, "Scenarios"),
        (8, "Compare"), (9, "Report")
    ]
    
    cols = st.columns(len(steps))
    completed = st.session_state.get("step_completed", {})
    
    for i, (step_num, label) in enumerate(steps):
        with cols[i]:
            is_completed = completed.get(step_num, False)
            is_current = step_num == current
            
            if is_completed:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; 
                                background: #27ae60; color: white; 
                                display: flex; align-items: center; justify-content: center; 
                                margin: 0 auto 4px; font-weight: bold;">✓</div>
                    <div style="font-size: 0.7rem; color: #27ae60; font-weight: 600;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            elif is_current:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; 
                                background: #3498db; color: white; 
                                display: flex; align-items: center; justify-content: center; 
                                margin: 0 auto 4px; font-weight: bold;">{step_num + 1}</div>
                    <div style="font-size: 0.7rem; color: #3498db; font-weight: 600;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; 
                                background: #ecf0f1; color: #bdc3c7; 
                                display: flex; align-items: center; justify-content: center; 
                                margin: 0 auto 4px; font-weight: bold;">{step_num + 1}</div>
                    <div style="font-size: 0.7rem; color: #95a5a6;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Connectors
            if i < len(steps) - 1:
                next_completed = completed.get(steps[i+1][0], False)
                connector_color = "#27ae60" if is_completed and next_completed else "#ecf0f1"
                st.markdown(f"""
                <div style="height: 2px; background: {connector_color}; margin-top: -26px; margin-bottom: 26px;"></div>
                """, unsafe_allow_html=True)


def format_years(value):
    """Format years for display."""
    if value is None or (isinstance(value, float) and (value != value or value == float('inf'))):
        return "> 20 years" if value == float('inf') else "N/A"
    return f"{value:.1f} years"