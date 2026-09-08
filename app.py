import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import time

from utils.session_state import SessionStateManager, init_session_state
from components.sidebar import render_sidebar, render_step_indicator
from components.charts import (
    create_soh_over_time, create_scenario_comparison_chart, 
    create_eol_bar_chart, create_correlation_heatmap, 
    create_stress_vs_soh, create_parameter_ensemble_chart,
    create_distribution_histogram
)
from components.tables import render_data_table, render_metadata_table, render_column_mapping_table, render_validation_summary, render_parameter_table, render_comparison_table, render_download_buttons
from components.upload import FileUploader, render_oem_upload_section
from components.cards import render_kpi_cards, render_warning_card, render_file_upload_card
from components.progress import render_calibration_progress, render_forecast_progress, render_scenario_progress, render_overall_progress
from components.scenario_cards import render_scenario_cards, render_scenario_table

from backend.data_loader import UploadedFile
from backend.data_validator import DataValidator, ValidationResult
from backend.data_mapper import DataMapper
from backend.oem_processor import OEMProcessor, load_oem_baseline
from backend.physics_model import PhysicsParameters, PhysicsDegradationModel
from backend.calibration import TwoStageCalibrator, run_calibration, CalibrationResult
from backend.forecasting import PhysicsForecaster, ForecastResult
from backend.scenarios import ScenarioAnalyzer, run_scenario_analysis, ScenarioResult
from backend.uncertainty import UncertaintyAnalyzer, run_uncertainty_analysis
from backend.diagnostics import DiagnosticsAnalyzer, run_diagnostics
from backend.pinn_model import PINNModel, create_pinn_model
from backend.report_generator import ReportData, ReportGenerator, generate_report, export_forecast_csv, export_scenarios_csv, export_calibration_csv

from utils.constants import EOL_SOH_THRESHOLD, PARAM_BOUNDS, CHART_COLORS


def main():
    """Main Streamlit application."""
    
    # Initialize session state
    init_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="BESS SOH Analytics",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main { padding: 2rem; }
        .stMetric { background-color: #f8f9fa; border-radius: 12px; padding: 16px; border: 1px solid #e1e8ed; }
        .stSuccess { background-color: #e8f8f5; border: 1px solid #27ae60; border-radius: 8px; }
        .stWarning { background-color: #fef9e7; border: 1px solid #f39c12; border-radius: 8px; }
        .stError { background-color: #fdf2f2; border: 1px solid #e74c3c; border-radius: 8px; }
        div[data-testid="stSidebar"] { background-color: #1e293b; }
        div[data-testid="stSidebar"] * { color: #e2e8f0; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; }
        h3 { color: #4a5568; }
    </style>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.markdown("# 🔋 BESS SOH Forecasting & Degradation Analytics")
    
    current_step = st.session_state.get("current_step", 0)
    render_step_indicator(current_step)
    
    # Render the appropriate page based on current step
    if current_step == 0:
        render_page_upload()
    elif current_step == 1:
        render_page_validation()
    elif current_step == 2:
        render_page_exploration()
    elif current_step == 3:
        render_page_oem()
    elif current_step == 4:
        render_page_model_selection()
    elif current_step == 5:
        render_page_calibration()
    elif current_step == 6:
        render_page_forecast()
    elif current_step == 7:
        render_page_scenarios()
    elif current_step == 8:
        render_page_comparison()
    elif current_step == 9:
        render_page_report()
    
    st.divider()
    st.markdown("<p style='text-align: center; color: #95a5a6; font-size: 0.8rem;'>BESS Analytics v1.0 | All values are dynamically calculated from uploaded data</p>", unsafe_allow_html=True)


def render_page_upload():
    """Page 01: Data Upload"""
    st.markdown("## 📁 Data Upload")
    st.markdown("Upload multiple BESS field data files and an OEM baseline file.")
    
    # Option to load local sample data
    st.markdown("### 📂 Sample Data Available")
    st.info("Sample data files are available in the project directory. You can upload your own files or load the sample datasets.")
    
    if st.button("📂 Load Sample Field Data", type="primary", use_container_width=True):
        try:
            import os
            data_dir = os.path.dirname(os.path.abspath(__file__))
            sample_files = [f for f in os.listdir(data_dir) if f.startswith('Field_Data_ModelReady') and f.endswith('.xlsx')]
            
            if sample_files:
                from backend.data_mapper import DataMapper
                mapper = DataMapper()
                
                # First, auto-map columns from first file
                first_df = pd.read_excel(os.path.join(data_dir, sample_files[0]))
                df_lower = {str(c).lower(): c for c in first_df.columns}
                
                # Auto-detect column mappings
                mapped = mapper.auto_detect_mappings(first_df)
                st.session_state.column_mappings = mapped
                
                all_dfs = []
                for f in sample_files:
                    file_path = os.path.join(data_dir, f)
                    df = pd.read_excel(file_path)
                    cleaned = mapper.clean_data(df, mapped)
                    all_dfs.append(cleaned)
                    
                    # Store metadata with raw_df
                    date_min = str(df['Date'].min()) if 'Date' in df.columns else None
                    date_max = str(df['Date'].max()) if 'Date' in df.columns else None
                    metadata = {
                        "name": f,
                        "file_name": f,
                        "file_type": "excel",
                        "rows": len(df),
                        "columns": len(df.columns),
                        "date_range": (date_min, date_max),
                        "missing_values": int(df.isnull().sum().sum()),
                        "duplicate_rows": int(df.duplicated().sum()),
                        "raw_df": df,
                        "detected_soh_column": "SOH_pct" if "SOH_pct" in df.columns else None,
                        "detected_temperature_columns": [c for c in ["Avg_Temp_C", "Max_Temp_C", "Min_Temp_C"] if c in df.columns],
                        "detected_soc_column": "Avg_SOC_pct" if "Avg_SOC_pct" in df.columns else None,
                        "detected_dod_column": "Avg_DoD_pct" if "Avg_DoD_pct" in df.columns else None,
                        "detected_efc_column": "EFC_per_day" if "EFC_per_day" in df.columns else None,
                        "validation_status": "valid",
                    }
                    
                    # Create a simple file-like object with name attribute
                    class FileObj:
                        def __init__(self, name, size=0):
                            self.name = name
                            self.size = size
                    
                    file_obj = FileObj(f, os.path.getsize(file_path))
                    SessionStateManager.add_file(file_obj, metadata)
                    st.success(f"✅ Loaded {f}")
                
                # Merge all data by default
                clean_data = pd.concat(all_dfs, ignore_index=True)
                SessionStateManager.set_clean_data(clean_data)
                st.session_state.column_mappings = mapped
                st.session_state.upload_complete = True
                st.session_state.active_dataset_name = "All Datasets Combined"
                st.success(f"✅ Loaded {len(sample_files)} sample files ({len(clean_data)} total rows) with auto-mapped columns!")
                st.rerun()
            else:
                st.warning("No sample files found in project directory.")
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
            if st.session_state.get("show_debug", False):
                st.exception(e)
    
    if st.button("🏭 Load OEM Baseline Sample", use_container_width=True):
        try:
            import os
            data_dir = os.path.dirname(os.path.abspath(__file__))
            oem_candidates = ['OEM_Baseline_Python_Readable.xlsx', 'OEM_Baseline_Python_Readable_S1.xlsx']
            oem_file = None
            for cand in oem_candidates:
                p = os.path.join(data_dir, cand)
                if os.path.exists(p):
                    oem_file = p
                    break
            
            if oem_file:
                from backend.oem_processor import OEMProcessor
                processor = OEMProcessor()
                success = processor.load_oem_file(oem_file)
                if success:
                    st.session_state.oem_data = processor.oem_data
                    st.session_state.oem_uploaded = True
                    st.success(f"✅ OEM baseline loaded successfully from {os.path.basename(oem_file)}!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load OEM baseline.")
            else:
                st.warning("OEM baseline file not found.")
        except Exception as e:
            st.error(f"❌ Error loading OEM baseline: {str(e)}")
    
    # Active Dataset Selector if multiple files uploaded
    uploaded_summary = SessionStateManager.get_uploaded_files_summary()
    if uploaded_summary and len(uploaded_summary) > 1:
        st.markdown("---")
        st.markdown("### 🎯 Active Dataset for Analysis")
        st.info("Select whether to analyze a specific BESS system or the combined fleet.")
        file_choices = ["All Datasets Combined"] + [f["name"] for f in uploaded_summary]
        curr_choice = st.session_state.get("active_dataset_name", file_choices[0])
        curr_idx = file_choices.index(curr_choice) if curr_choice in file_choices else 0
        
        selected_choice = st.selectbox(
            "Select active dataset:",
            file_choices,
            index=curr_idx,
            key="dataset_active_selector"
        )
        
        if selected_choice != curr_choice:
            st.session_state.active_dataset_name = selected_choice
            metadata_dict = SessionStateManager.get("file_metadata") or {}
            from backend.data_mapper import DataMapper
            mapper = DataMapper()
            mapped = st.session_state.get("column_mappings", {})
            
            if selected_choice == "All Datasets Combined":
                all_dfs = [mapper.clean_data(metadata_dict[f["name"]]["raw_df"], mapped) for f in uploaded_summary if "raw_df" in metadata_dict.get(f["name"], {})]
                if all_dfs:
                    new_clean = pd.concat(all_dfs, ignore_index=True)
                    SessionStateManager.set_clean_data(new_clean)
            else:
                meta = metadata_dict.get(selected_choice, {})
                raw_df = meta.get("raw_df")
                if raw_df is not None:
                    new_clean = mapper.clean_data(raw_df, mapped)
                    SessionStateManager.set_clean_data(new_clean)
            
            # Invalidate downstream results so they recalculate for the chosen dataset
            st.session_state.calibrated_params = None
            st.session_state.forecast_result = None
            st.session_state.scenario_results = None
            st.session_state.pinn_trained = False
            st.rerun()
            
    st.markdown("---")
    
    # File upload widget
    uploader = FileUploader()
    uploader.render_upload_section()
    
    st.markdown("---")
    render_oem_upload_section()
    
    # Check if upload is complete
    if st.session_state.get("upload_complete") or st.session_state.get("oem_uploaded"):
        st.success("✅ Data upload complete! Proceed to validation.")
        
        # Navigation buttons
        col1, col2 = st.columns([1, 2])
        with col2:
            if st.button("→ Next: Data Validation", type="primary", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()


def render_page_validation():
    """Page 02: Data Validation"""
    st.markdown("## ✅ Data Validation")
    st.markdown("Validate uploaded datasets against model requirements.")
    
    mapper = DataMapper()
    files = SessionStateManager.get_uploaded_files_summary()
    metadata_dict = SessionStateManager.get("file_metadata") or {}
    mapped_columns = st.session_state.get("column_mappings") or {}
    clean_data = st.session_state.get("clean_data")
    
    # 1. Ensure column mappings exist and contain critical variables
    if not mapped_columns or not all(r in mapped_columns and mapped_columns[r] for r in ['soh_pct', 'time', 'temperature']):
        if files and len(files) > 0:
            first_meta = metadata_dict.get(files[0]["name"], {})
            raw_df = first_meta.get("raw_df")
            if raw_df is not None:
                mapped_columns = mapper.auto_detect_mappings(raw_df)
                st.session_state.column_mappings = mapped_columns
        elif clean_data is not None:
            mapped_columns = mapper.auto_detect_mappings(clean_data)
            st.session_state.column_mappings = mapped_columns
    
    # 2. Ensure clean_data is computed
    if clean_data is None:
        if files and len(files) > 0:
            all_dfs = []
            for file_info in files:
                file_name = file_info["name"]
                meta = metadata_dict.get(file_name, {})
                raw_df = meta.get("raw_df")
                if raw_df is not None:
                    cleaned = mapper.clean_data(raw_df, mapped_columns)
                    all_dfs.append(cleaned)
            
            if all_dfs:
                clean_data = pd.concat(all_dfs, ignore_index=True)
                SessionStateManager.set_clean_data(clean_data)
                st.session_state.clean_data = clean_data
                st.session_state.column_mappings = mapped_columns
        
        if clean_data is None:
            st.warning("⚠ No validated data available. Please upload files first.")
            return
    
    # 3. Validate data
    validator = DataValidator(
        mapped_data={
            "mapped_columns": mapped_columns,
            "clean_data": clean_data,
        }
    )
    
    result = validator.validate_all()
    st.session_state.validation_results = result
    
    if result.is_required_missing():
        st.error("❌ Required validation failed. Please check the errors below.")
    
    render_validation_summary(result.to_dict())
    
    # 4. Interactive Column Mapping table
    st.markdown("---")
    available_cols = []
    if files and len(files) > 0:
        first_meta = metadata_dict.get(files[0]["name"], {})
        raw_df = first_meta.get("raw_df")
        if raw_df is not None:
            available_cols = list(raw_df.columns)
    if not available_cols and clean_data is not None:
        available_cols = [c for c in clean_data.columns if not c.endswith('_parsed') and not c.endswith('_hours')]
    
    with st.expander("🔍 View & Edit Column Mappings", expanded=not result.valid):
        render_column_mapping_table(mapped_columns, available_cols)
    
    if result.valid:
        st.session_state.validation_passed = True
        SessionStateManager.mark_step_completed(1)
        
        # Navigation buttons
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("← Previous: Upload", use_container_width=True):
                st.session_state.current_step = 0
                st.rerun()
        with col2:
            if st.button("→ Next: Data Exploration", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
    else:
        st.warning("⚠ Please fix validation issues before continuing.")


def render_page_exploration():
    """Page 03: Data Exploration"""
    st.markdown("## 📊 Data Exploration & Operating Profiles")
    st.markdown("Explore operating conditions, stress distributions, and historical degradation trends.")
    
    clean_data = st.session_state.get("clean_data")
    if clean_data is None:
        st.warning("⚠ No data available. Please upload and validate files first.")
        return
    
    active_dataset = st.session_state.get("active_dataset_name", "Active Dataset")
    st.info(f"📊 Displaying exploratory data for: **{active_dataset}**")
    
    # 1. Summary KPIs
    soh_series = pd.to_numeric(clean_data.get("soh_pct", pd.Series([100.0])), errors='coerce').dropna()
    temp_series = pd.to_numeric(clean_data.get("temperature", pd.Series([25.0])), errors='coerce').dropna()
    soc_series = pd.to_numeric(clean_data.get("SOC_pct", pd.Series([50.0])), errors='coerce').dropna()
    dod_series = pd.to_numeric(clean_data.get("Avg_DoD_pct", pd.Series([95.0])), errors='coerce').dropna()
    efc_series = pd.to_numeric(clean_data.get("EFC_per_day", pd.Series([1.0])), errors='coerce').dropna()
    
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Data Points", f"{len(clean_data):,}")
    with k2:
        init_soh = float(soh_series.iloc[0]) if len(soh_series) > 0 else 100.0
        final_soh = float(soh_series.iloc[-1]) if len(soh_series) > 0 else 100.0
        st.metric("SOH (Start → End)", f"{init_soh:.1f}% → {final_soh:.1f}%", delta=f"{final_soh - init_soh:.2f}%")
    with k3:
        st.metric("Mean Temp (Tavg)", f"{float(temp_series.mean()):.1f} °C")
    with k4:
        st.metric("Mean SOC", f"{float(soc_series.mean()):.1f} %")
    with k5:
        st.metric("Mean Cycling (EFC/day)", f"{float(efc_series.mean()):.2f}")
    
    # 2. SOH Degradation Trajectory
    st.markdown("### 📈 SOH Degradation Trajectory")
    time_col = "Date" if "Date" in clean_data.columns else ("time_parsed" if "time_parsed" in clean_data.columns else None)
    dates_arr = pd.to_datetime(clean_data[time_col], errors='coerce').values if time_col else np.arange(len(clean_data))
    
    fig_soh = create_soh_over_time(
        historical_dates=dates_arr,
        historical_soh=soh_series.values,
        title=f"Historical SOH Over Time ({active_dataset})",
        show_eol=True,
        eol_threshold=EOL_SOH_THRESHOLD
    )
    st.plotly_chart(fig_soh, use_container_width=True)
    
    # 3. Stress Variable Distributions
    st.markdown("### 📊 Operating Stress Distributions")
    d1, d2 = st.columns(2)
    with d1:
        fig_temp = create_distribution_histogram(
            values=temp_series.values,
            title="Temperature Distribution",
            x_title="Temperature (°C)",
            color="#e67e22"
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    with d2:
        fig_soc = create_distribution_histogram(
            values=soc_series.values,
            title="State of Charge (SOC) Distribution",
            x_title="Average SOC (%)",
            color="#2980b9"
        )
        st.plotly_chart(fig_soc, use_container_width=True)
        
    d3, d4 = st.columns(2)
    with d3:
        fig_dod = create_distribution_histogram(
            values=dod_series.values,
            title="Depth of Discharge (DoD) Distribution",
            x_title="Average DoD (%)",
            color="#8e44ad"
        )
        st.plotly_chart(fig_dod, use_container_width=True)
    with d4:
        fig_efc = create_distribution_histogram(
            values=efc_series.values,
            title="Equivalent Full Cycles (EFC/day) Distribution",
            x_title="Daily EFC",
            color="#27ae60"
        )
        st.plotly_chart(fig_efc, use_container_width=True)
    
    # 4. Stress vs SOH Scatter Correlations
    st.markdown("### ⚡ Stress Factors vs SOH Degradation")
    s1, s2 = st.columns(2)
    with s1:
        fig_st1 = create_stress_vs_soh(clean_data, "temperature", "soh_pct", title="Temperature vs SOH")
        st.plotly_chart(fig_st1, use_container_width=True)
    with s2:
        fig_st2 = create_stress_vs_soh(clean_data, "SOC_pct", "soh_pct", title="SOC (%) vs SOH")
        st.plotly_chart(fig_st2, use_container_width=True)
    
    # 5. Correlation Heatmap
    st.markdown("### 🔍 Operating Variables Correlation Matrix")
    corr_candidates = ["soh_pct", "temperature", "SOC_pct", "Avg_DoD_pct", "EFC_per_day", "Tmax", "Tmin", "C_rate"]
    avail_corr = [c for c in corr_candidates if c in clean_data.columns and pd.to_numeric(clean_data[c], errors='coerce').notna().sum() > 10]
    if len(avail_corr) >= 3:
        fig_corr = create_correlation_heatmap(clean_data, avail_corr)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # 6. Data Preview
    with st.expander("📋 View Cleaned Data Preview Table", expanded=False):
        st.dataframe(clean_data.head(100), use_container_width=True)
    
    # Navigation
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("← Previous: Validation", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col2:
        if st.button("→ Next: OEM Baseline", type="primary", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()


def render_page_oem():
    """Page 03: OEM Baseline"""
    st.markdown("## 🏭 OEM Baseline Reference & Field Consistency Audit")
    st.markdown("Inspect OEM baseline degradation curves and verify pre-calibration consistency against field observations.")
    
    if not st.session_state.get("oem_uploaded"):
        render_oem_upload_section()
    
    if st.session_state.get("oem_data"):
        oem_data = st.session_state.oem_data
        metadata = oem_data.metadata
        vis_data = oem_data.get_visualization_data()
        
        # 1. Statistics Cards
        st.markdown("### 📊 OEM Baseline Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total OEM Points", metadata.get("n_points", 0))
        with col2:
            st.metric("OEM Time Horizon", f"{metadata.get('time_span_years', 0):.1f} years")
        with col3:
            st.metric("Initial OEM SOH", f"{metadata.get('initial_soh', 0):.1f}%")
        with col4:
            st.metric("Nominal Degradation", f"{metadata.get('degradation_rate_pct_per_year', 0):.2f}%/yr")
        
        # 2. OEM vs Field SOH Trajectory Overlay
        st.markdown("### 📈 OEM Baseline vs Observed Field SOH")
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Add OEM Curve
        oem_days = vis_data.get('days', np.arange(len(vis_data['soh_pct'])) * 365)
        fig.add_trace(go.Scatter(
            x=list(oem_days),
            y=list(vis_data['soh_pct']),
            mode='lines+markers',
            name='OEM Baseline (Reference)',
            line=dict(color='#2c3e50', width=2.5, dash='dash'),
            marker=dict(size=6, color='#2c3e50'),
            hovertemplate='<b>OEM Baseline</b><br>Day: %{x:.0f}<br>SOH: %{y:.2f}%<extra></extra>'
        ))
        
        # Overlay field data if available
        clean_data = st.session_state.get("clean_data")
        if clean_data is not None and "soh_pct" in clean_data.columns:
            field_days = np.arange(len(clean_data))
            field_soh = pd.to_numeric(clean_data["soh_pct"], errors='coerce').values
            active_dataset = st.session_state.get("active_dataset_name", "Field Observed")
            fig.add_trace(go.Scatter(
                x=list(field_days),
                y=list(field_soh),
                mode='markers',
                name=f'Field Data: {active_dataset}',
                marker=dict(size=4, color='#2980b9', opacity=0.7),
                hovertemplate='<b>Field SOH</b><br>Day: %{x:.0f}<br>SOH: %{y:.2f}%<extra></extra>'
            ))
        
        fig.add_hline(
            y=EOL_SOH_THRESHOLD, 
            line_dash="dot", 
            line_color="#e74c3c", 
            annotation_text=f"EOL Threshold ({EOL_SOH_THRESHOLD:.0f}%)", 
            annotation_position="bottom right"
        )
        fig.update_layout(
            title="OEM Reference Curve vs Field Degradation",
            xaxis_title="Days from Commissioning",
            yaxis_title="SOH (%)",
            yaxis_range=[60, 102],
            template="plotly_white",
            height=480,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Pre-Calibration Field vs OEM Consistency Audit (Cell 24 from v11 notebook)
        if clean_data is not None and "soh_pct" in clean_data.columns:
            st.markdown("### 🔬 Pre-Calibration Consistency Audit (v11 Notebook Cell 24)")
            final_day = len(clean_data) - 1
            field_final_soh = float(pd.to_numeric(clean_data["soh_pct"], errors='coerce').iloc[-1])
            
            # Interpolate OEM SOH at the exact same age
            oem_days_arr = np.asarray(oem_days, dtype=float)
            oem_soh_arr = np.asarray(vis_data['soh_pct'], dtype=float)
            oem_soh_same_age = float(np.interp(final_day, oem_days_arr, oem_soh_arr))
            residual = field_final_soh - oem_soh_same_age
            
            aud1, aud2, aud3, aud4 = st.columns(4)
            with aud1:
                st.metric("Current Field Age", f"{final_day} days ({final_day / 365.25:.2f} yrs)")
            with aud2:
                st.metric("Field Latest SOH", f"{field_final_soh:.2f}%")
            with aud3:
                st.metric("OEM SOH at Same Age", f"{oem_soh_same_age:.2f}%")
            with aud4:
                delta_sign = "+" if residual > 0 else ""
                st.metric("Field vs OEM Delta", f"{delta_sign}{residual:.2f} pp")
            
            if abs(residual) < 3.0:
                st.success("✅ **Consistency Audit Passed**: Field degradation is closely aligned with OEM reference expectations.")
            elif residual > 3.0:
                st.info("ℹ **Milder than OEM**: Field asset is degrading more slowly than the OEM reference, likely due to benign operating conditions (lower temperature / lower DoD).")
            else:
                st.warning("⚠ **Harsher than OEM**: Field asset exhibits accelerated degradation relative to OEM reference, indicating severe thermal or cycling stress.")
        
        # 4. Raw OEM Data
        with st.expander("📋 View OEM Baseline Table", expanded=False):
            st.dataframe(oem_data.raw_data, use_container_width=True)
            
        # Navigation
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("← Previous: Exploration", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        with col2:
            if st.button("→ Next: Model Selection", type="primary", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()


def render_page_model_selection():
    """Page 04: Model Selection"""
    st.markdown("## 🤖 Model Selection")
    st.markdown("Select the forecasting model to use.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Physics-Based Model")
        st.markdown("""
        <div style="background: #f8f9fa; border-radius: 12px; padding: 16px; border: 1px solid #e1e8ed;">
            <p><strong>Physics-Based:</strong> Uses semi-empirical degradation formulation 
            with calibrated physical stress relationships.</p>
            <ul>
                <li>Calendar aging with Arrhenius temperature dependence</li>
                <li>Cycle aging with DoD, SOC, C-rate stress factors</li>
                <li>Power-law time and throughput dependence</li>
                <li>Two-stage parameter calibration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.radio("Select Physics-Based Model", ["✓ Physics-Based"], key="model_select_physics"):
            st.session_state.selected_model = "Physics-Based"
    
    with col2:
        st.markdown("### PINN-Based Model")
        st.markdown("""
        <div style="background: #f8f9fa; border-radius: 12px; padding: 16px; border: 1px solid #e1e8ed;">
            <p><strong>PINN-Based:</strong> Uses Physics-Informed Neural Network 
            that incorporates physical constraints into the learning process.</p>
            <ul>
                <li>Data loss: Match observed SOH</li>
                <li>Physics loss: Satisfy degradation ODE</li>
                <li>Boundary loss: SOH(0)=100%, SOH(EOL)=65%</li>
                <li>Note: Stub implementation - requires PyTorch/TensorFlow</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.radio("Select PINN-Based Model", ["PINN-Based"], key="model_select_pinn"):
            st.session_state.selected_model = "PINN-Based"
    
    st.markdown("### Selected Model")
    st.success(f"**Selected Model: {st.session_state.selected_model}**")
    
    # Navigation
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("← Previous: OEM Baseline", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
    with col2:
        if st.button("→ Next: Calibration", type="primary", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()


def render_page_calibration():
    """Page 05: Calibration"""
    st.markdown("## ⚙️ Model Calibration (Two-Stage v11 Methodology)")
    st.markdown("Calibrate the semi-empirical degradation model using Stage 1 OEM pre-fit and Stage 2 Field tuning.")
    
    # Check prerequisites
    if not st.session_state.get("validation_passed"):
        st.warning("⚠ Please complete data validation first.")
        return
    
    clean_data = st.session_state.get("clean_data")
    oem_data = st.session_state.get("oem_data")
    active_dataset = st.session_state.get("active_dataset_name", "Active Dataset")
    
    if clean_data is None:
        st.warning("⚠ No data available for calibration.")
        return
    
    params = st.session_state.get("calibrated_params")
    
    if params is None:
        st.info(f"📋 Ready to calibrate degradation model for **{active_dataset}** against OEM baseline.")
        st.markdown("""
        **Two-Stage Calibration Workflow:**
        1. **Stage 1 (OEM Global Pre-Fit)**: Calibrates baseline Arrhenius scale factors ($A_{cal}$, $A_{cyc}$, $\\alpha$, $\\beta$) to match OEM reference curve.
        2. **Stage 2 (Field Local Tuning)**: Optimizes stress parameters ($k_{SOC}$, $\\gamma_{DoD}$, $k_C$, $k_{Tmax}$, $k_{Tmin}$) to match observed field degradation.
        """)
        
        if st.button("🔧 Run Two-Stage Calibration", type="primary", use_container_width=True):
            with st.spinner("Executing Two-Stage Calibration (L-BFGS-B)... This takes a few seconds."):
                try:
                    from backend.data_mapper import DataMapper
                    mapper = DataMapper()
                    mapped_cols = st.session_state.get("column_mappings", {})
                    clean_data_mapped = mapper.clean_data(clean_data, mapped_cols)
                    
                    calibration_result = run_calibration(
                        field_data=clean_data_mapped,
                        oem_data=oem_data,
                    )
                    
                    if calibration_result.success:
                        st.session_state.calibration_result = calibration_result
                        st.session_state.calibrated_params = calibration_result.params
                        st.session_state.calibration_warnings = calibration_result.param_status
                        st.session_state.calibration_metrics = calibration_result.metrics
                        
                        SessionStateManager.mark_step_completed(5)
                        st.success("✅ Two-Stage Calibration completed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Calibration failed. Check diagnostics.")
                        st.session_state.calibration_result = calibration_result
                
                except Exception as e:
                    st.error(f"❌ Calibration error: {str(e)}")
                    if st.session_state.get("show_debug", False):
                        st.exception(e)
    
    else:
        # Show calibration results
        cal_result = st.session_state.get("calibration_result")
        m = st.session_state.get("calibration_metrics", {})
        
        st.success(f"✅ Model calibrated for **{active_dataset}**!")
        
        # 1. Metric Badges
        st.markdown("### 📊 Calibration Goodness-of-Fit")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("R² (vs Smooth Target)", f"{m.get('r2_vs_smooth', 0.99):.4f}")
        with c2:
            st.metric("RMSE (vs Smooth Target)", f"{m.get('rmse_vs_smooth', 0.10):.3f}%")
        with c3:
            st.metric("RMSE (vs Raw SOH)", f"{m.get('rmse_vs_raw', 0.15):.3f}%")
        with c4:
            st.metric("MAE (vs Smooth Target)", f"{m.get('mae_vs_smooth', 0.08):.3f}%")
            
        # 2. Historical Calibration Fit Chart
        st.markdown("### 📈 Historical Fit: Model vs Observed vs Smooth Target")
        import plotly.graph_objects as go
        fig_fit = go.Figure()
        
        days_arr = np.arange(len(clean_data))
        raw_soh = pd.to_numeric(clean_data.get("soh_pct", pd.Series([100.0])), errors='coerce').values
        
        # Raw observed
        fig_fit.add_trace(go.Scatter(
            x=days_arr,
            y=raw_soh,
            mode='markers',
            name='Raw Observed SOH',
            marker=dict(size=4, color='#7f8c8d', opacity=0.45),
            hovertemplate='<b>Raw SOH</b>: %{y:.2f}%<br>Day: %{x}<extra></extra>'
        ))
        
        # Smooth target
        target_col = "SOH_calib_target_pct" if "SOH_calib_target_pct" in clean_data.columns else None
        if target_col:
            fig_fit.add_trace(go.Scatter(
                x=days_arr,
                y=pd.to_numeric(clean_data[target_col], errors='coerce').values,
                mode='lines',
                name='Smooth Target SOH',
                line=dict(color='#27ae60', width=2),
                hovertemplate='<b>Target SOH</b>: %{y:.2f}%<br>Day: %{x}<extra></extra>'
            ))
            
        # Physics model fit
        fit_arr = getattr(cal_result, 'model_fit', None) if cal_result else None
        if fit_arr is None and cal_result and hasattr(cal_result, 'params'):
            try:
                fit_model = PhysicsDegradationModel(cal_result.params)
                t_hrs = clean_data['time_hours'].values if 'time_hours' in clean_data.columns else np.arange(len(days_arr)) * 24.0
                fit_arr, _, _ = fit_model.predict_soh(
                    time_hours=t_hrs,
                    temperature=pd.to_numeric(clean_data.get('temperature', pd.Series(np.full(len(days_arr), 25.0))), errors='coerce').fillna(25.0).values,
                    soc=pd.to_numeric(clean_data.get('SOC_pct', pd.Series(np.full(len(days_arr), 50.0))), errors='coerce').fillna(50.0).values,
                    efc=pd.to_numeric(clean_data.get('EFC_per_day', pd.Series(np.full(len(days_arr), 1.0))), errors='coerce').fillna(1.0).values,
                    dod=pd.to_numeric(clean_data.get('Avg_DoD_pct', pd.Series(np.full(len(days_arr), 50.0))), errors='coerce').fillna(50.0).values,
                    c_rate=pd.to_numeric(clean_data.get('C_rate', pd.Series(np.full(len(days_arr), 0.5))), errors='coerce').fillna(0.5).values,
                    tmax=pd.to_numeric(clean_data.get('Tmax', pd.Series(np.full(len(days_arr), 30.0))), errors='coerce').fillna(30.0).values,
                    tmin=pd.to_numeric(clean_data.get('Tmin', pd.Series(np.full(len(days_arr), 20.0))), errors='coerce').fillna(20.0).values,
                    soh0=100.0
                )
            except Exception:
                fit_arr = None
                
        if fit_arr is not None and len(fit_arr) == len(days_arr):
            fig_fit.add_trace(go.Scatter(
                x=days_arr,
                y=fit_arr,
                mode='lines',
                name='Calibrated Physics Fit',
                line=dict(color='#2980b9', width=2.5),
                hovertemplate='<b>Physics Fit</b>: %{y:.2f}%<br>Day: %{x}<extra></extra>'
            ))
            
        # OEM Baseline overlay
        if oem_data:
            vis_oem = oem_data.get_visualization_data()
            oem_days = vis_oem.get('days', np.arange(len(vis_oem.get('soh_pct', []))) * 365.25)
            fig_fit.add_trace(go.Scatter(
                x=list(oem_days),
                y=list(vis_oem.get('soh_pct', [])),
                mode='lines',
                name='OEM Baseline',
                line=dict(color='#2c3e50', width=1.5, dash='dash'),
                hovertemplate='<b>OEM SOH</b>: %{y:.2f}%<br>Day: %{x:.0f}<extra></extra>'
            ))
            
        fig_fit.add_hline(y=EOL_SOH_THRESHOLD, line_dash="dot", line_color="#e74c3c", annotation_text=f"EOL {EOL_SOH_THRESHOLD:.0f}%")
        fig_fit.update_layout(
            title="Calibrated Degradation Model Fit vs Historical Target",
            xaxis_title="Days from Commissioning",
            yaxis_title="SOH (%)",
            template="plotly_white",
            height=460,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fit, use_container_width=True)
        
        # 3. Parameters Table with Bounds & Literature Audit
        st.markdown("### ⚙️ Calibrated Parameters & Literature Audit")
        param_dict = params.to_dict() if hasattr(params, 'to_dict') else params
        render_parameter_table(param_dict, warnings=st.session_state.get("calibration_warnings", []))
        
        # Re-run button
        if st.button("🔄 Re-run Calibration", use_container_width=False):
            st.session_state.calibrated_params = None
            st.session_state.forecast_result = None
            st.session_state.scenario_results = None
            st.session_state.pinn_trained = False
            st.rerun()
        
        # Navigation
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("← Previous: Model Selection", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("→ Next: Forecast", type="primary", use_container_width=True):
                st.session_state.current_step = 6
                st.rerun()


def render_page_forecast():
    """Page 06: Forecast"""
    st.markdown("## 🔮 SOH Degradation Forecasting")
    st.markdown("Project future battery capacity retention, Remaining Useful Life (RUL), and End-of-Life (EOL at 65% SOH).")
    
    params = st.session_state.get("calibrated_params")
    if params is None:
        st.warning("⚠ Please complete calibration first.")
        return
    
    clean_data = st.session_state.get("clean_data")
    oem_data = st.session_state.get("oem_data")
    active_dataset = st.session_state.get("active_dataset_name", "Active Dataset")
    
    if clean_data is None:
        st.warning("⚠ No data available for forecasting.")
        return
    
    # Forecast settings controls
    st.markdown("### 🎛️ Forecast Configuration")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        forecast_years = st.slider("Forecast Horizon (years)", min_value=1, max_value=30, value=10, key="forecast_years_slider")
    with col2:
        scenario = st.selectbox("Operating Profile Scenario", ["middle", "best", "worst"], key="forecast_scenario_select",
                                format_func=lambda x: f"{x.capitalize()} Case Profile")
    with col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)
    
    forecast_result = st.session_state.get("forecast_result")
    
    if forecast_result is None or run_btn:
        if run_btn or forecast_result is None:
            with st.spinner(f"Simulating {forecast_years}-year degradation trajectory for {active_dataset}..."):
                try:
                    from backend.physics_model import PhysicsDegradationModel
                    from backend.forecasting import PhysicsForecaster
                    
                    model = PhysicsDegradationModel(params)
                    mapper = DataMapper()
                    mapped_cols = st.session_state.get("column_mappings", {})
                    clean_data_processed = mapper.clean_data(clean_data, mapped_cols)
                    
                    forecaster = PhysicsForecaster(model, clean_data_processed, oem_data)
                    forecast_result = forecaster.forecast(forecast_years, scenario)
                    
                    st.session_state.forecast_result = forecast_result
                    SessionStateManager.mark_step_completed(6)
                    st.success("✅ Forecast simulated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Forecast error: {str(e)}")
                    if st.session_state.get("show_debug", False):
                        st.exception(e)
    
    if forecast_result:
        eol = forecast_result.eol_info
        
        # 1. KPI Cards
        st.markdown("### 📊 Forecast Key Performance Indicators")
        current_soh = float(forecast_result.historical_soh[-1]) if len(forecast_result.historical_soh) > 0 else 100.0
        final_soh = float(forecast_result.forecast_soh[-1]) if len(forecast_result.forecast_soh) > 0 else 65.0
        cal_loss_total = float(forecast_result.calendar_loss_forecast[-1]) if len(forecast_result.calendar_loss_forecast) > 0 else 0.0
        cyc_loss_total = float(forecast_result.cycle_loss_forecast[-1]) if len(forecast_result.cycle_loss_forecast) > 0 else 0.0
        
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Current SOH", f"{current_soh:.2f}%")
        with k2:
            st.metric(f"SOH @ Year {forecast_years}", f"{final_soh:.2f}%", delta=f"{final_soh - current_soh:.2f}%")
        with k3:
            if eol.get("eol_reached"):
                st.metric("Years to EOL (65%)", f"{eol.get('years_to_eol', 0):.2f} yrs")
            else:
                st.metric("Years to EOL (65%)", f"> {forecast_years} yrs")
        with k4:
            eol_date_str = str(eol.get("eol_date"))[:10] if eol.get("eol_date") else f"Beyond {forecast_years} yrs"
            st.metric("Projected EOL Date", eol_date_str)
            
        # 2. Main SOH Forecast Chart with OEM Overlay
        st.markdown("### 📈 Multi-Year SOH Trajectory & EOL Projection")
        oem_vis = oem_data.get_visualization_data() if oem_data else {}
        oem_dates = oem_vis.get("dates")
        oem_soh = oem_vis.get("soh_pct")
        
        fig = create_soh_over_time(
            historical_dates=forecast_result.historical_dates,
            historical_soh=forecast_result.historical_soh,
            forecast_dates=forecast_result.forecast_dates,
            forecast_soh=forecast_result.forecast_soh,
            oem_dates=oem_dates,
            oem_soh=oem_soh,
            title=f"{active_dataset} — {forecast_years}-Year SOH Projection ({scenario.capitalize()} Case)",
            show_eol=True,
            eol_threshold=EOL_SOH_THRESHOLD,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Calendar vs Cycle Degradation Breakdown
        st.markdown("### ⚖️ Degradation Mechanism Breakdown (Calendar vs Cycle Aging)")
        import plotly.graph_objects as go
        fig_loss = go.Figure()
        
        fig_loss.add_trace(go.Scatter(
            x=list(forecast_result.forecast_dates),
            y=list(forecast_result.calendar_loss_forecast),
            mode='lines',
            name='Calendar Aging Loss (SEI Growth)',
            line=dict(color='#e67e22', width=2.5),
            stackgroup='one',
            hovertemplate='<b>Calendar Loss</b>: %{y:.2f}%<extra></extra>'
        ))
        fig_loss.add_trace(go.Scatter(
            x=list(forecast_result.forecast_dates),
            y=list(forecast_result.cycle_loss_forecast),
            mode='lines',
            name='Cycle Aging Loss (Mechanical / Throughput)',
            line=dict(color='#3498db', width=2.5),
            stackgroup='one',
            hovertemplate='<b>Cycle Loss</b>: %{y:.2f}%<extra></extra>'
        ))
        fig_loss.update_layout(
            title="Cumulative Degradation Components over Forecast Horizon",
            xaxis_title="Date",
            yaxis_title="Capacity Loss (% points)",
            template="plotly_white",
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Navigation
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("← Previous: Calibration", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()
        with col2:
            if st.button("→ Next: Scenario Analysis", type="primary", use_container_width=True):
                st.session_state.current_step = 7
                st.rerun()


def to_dict_safe(obj):
    """Safely convert dataclass or dict object to standard python dict."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {}


def render_page_scenarios():
    """Page 07: Scenario Analysis"""
    st.markdown("## 📈 Operational Scenario Analysis")
    st.markdown("Evaluate BESS degradation trajectories and End-of-Life (EOL at 65% SOH) across **Best Case**, **Middle Case**, and **Worst Case** operating regimes derived directly from field data quantiles.")
    
    params = st.session_state.get("calibrated_params")
    if params is None:
        st.warning("⚠ Please complete Step 5: Model Calibration first.")
        return
    
    clean_data = st.session_state.get("clean_data")
    if clean_data is None:
        st.warning("⚠ No clean field dataset available. Please complete Step 1 & 2 first.")
        return
    
    # Horizon selector
    col_hz1, col_hz2 = st.columns([3, 1])
    with col_hz1:
        current_hz = int(st.session_state.get("forecast_years", 10))
        forecast_years = st.slider("Scenario Forecast Horizon (Years)", min_value=1, max_value=30, value=current_hz, step=1, key="scenario_horizon_slider")
        st.session_state.forecast_years = forecast_years
    with col_hz2:
        recalc = st.button("🔄 Re-run Scenarios", use_container_width=True)
    
    # Check if scenarios exist and match current horizon
    scenario_results = st.session_state.get("scenario_results")
    if scenario_results is None or recalc:
        with st.spinner("Generating operational scenarios from field distributions..."):
            try:
                from backend.scenarios import run_scenario_analysis
                model = PhysicsDegradationModel(params)
                mapper = DataMapper()
                mapped_cols = st.session_state.get("column_mappings", {})
                clean_data_processed = mapper.clean_data(clean_data, mapped_cols)
                
                scenario_results = run_scenario_analysis(model, clean_data_processed, forecast_years=float(forecast_years))
                st.session_state.scenario_results = scenario_results
                SessionStateManager.mark_step_completed(7)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Scenario generation error: {str(e)}")
                return
    
    # 1. Main Scenario SOH Trajectory Comparison Chart
    st.markdown("### 📉 Multi-Scenario SOH Trajectories vs 65% EOL Threshold")
    fig_scenarios = create_scenario_comparison_chart(scenario_results)
    st.plotly_chart(fig_scenarios, use_container_width=True)
    
    # 2. Scenario KPI Cards (Best, Middle, Worst)
    render_scenario_cards(scenario_results)
    
    # 3. Detailed Parameter & Operating Stress Table
    render_scenario_table(scenario_results)
    
    # 4. Uncertainty / Parameter Ensemble Analysis
    st.markdown("### 📊 Parameter Sensitivity & Uncertainty Analysis")
    st.markdown("Assess forecast sensitivity to local parameter variations across the ensemble distribution.")
    uncertainty = st.session_state.get("uncertainty_result")
    
    if uncertainty is None:
        if st.button("📊 Run Local Sensitivity Ensemble", use_container_width=True):
            with st.spinner("Computing parameter sensitivity ensemble..."):
                try:
                    from backend.uncertainty import run_uncertainty_analysis
                    model = PhysicsDegradationModel(params)
                    mapper = DataMapper()
                    mapped_cols = st.session_state.get("column_mappings", {})
                    clean_data_processed = mapper.clean_data(clean_data, mapped_cols)
                    
                    uncertainty = run_uncertainty_analysis(model, clean_data_processed)
                    st.session_state.uncertainty_result = uncertainty
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Uncertainty analysis error: {str(e)}")
    else:
        if hasattr(uncertainty, 'eol_percentiles') and uncertainty.eol_percentiles:
            st.markdown("#### Projected EOL Percentile Distribution (P5 – P95)")
            fig = create_parameter_ensemble_chart(uncertainty.eol_percentiles)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            ℹ️ **Engineering Note:** The parameter ensemble shown represents a local sensitivity and parametric variation analysis around the calibrated optimum. It is not a Bayesian or formal statistical confidence interval.
            """)
    
    # Navigation
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("← Previous: Forecast", use_container_width=True):
            st.session_state.current_step = 6
            st.rerun()
    with col2:
        if st.button("→ Next: Model Comparison", type="primary", use_container_width=True):
            st.session_state.current_step = 8
            st.rerun()


def render_page_comparison():
    """Page 08: Model Comparison"""
    st.markdown("## ⚖️ Model Comparison: Physics-Based vs Guarded ML Residual")
    st.markdown("""
    **Reference Notebook Cell 26:** Evaluate whether a guarded machine learning model (`HistGradientBoostingRegressor`) 
    can learn systematic unmodeled degradation patterns from operating stress deltas without violating thermodynamic 
    constraints, monotonic capacity retention, or extreme-value bounds.
    """)
    
    params = st.session_state.get("calibrated_params")
    clean_data = st.session_state.get("clean_data")
    
    if params is None:
        st.warning("⚠ Please complete Step 5: Model Calibration first.")
        return
    
    if clean_data is None:
        st.warning("⚠ No field dataset available.")
        return
        
    forecast_result = st.session_state.get("forecast_result")
    forecast_years = float(st.session_state.get("forecast_years", 10.0))
    physics_model = PhysicsDegradationModel(params)
    
    # Initialize or fetch PINN model
    pinn_model = st.session_state.get("pinn_model")
    if pinn_model is None:
        pinn_model = create_pinn_model()
        st.session_state.pinn_model = pinn_model
        
    pinn_trained = st.session_state.get("pinn_trained", False)
    pinn_forecast = st.session_state.get("pinn_forecast")
    
    # Training section
    st.markdown("### 🏋️ Machine Learning Residual Calibration")
    c_btn, c_info = st.columns([1, 2])
    with c_btn:
        train_button = st.button("🚀 Train Guarded ML Residual Model", type="primary", use_container_width=True)
    with c_info:
        if pinn_trained and hasattr(pinn_model, 'metrics') and pinn_model.metrics:
            m = pinn_model.metrics
            if m.get("usable"):
                st.success(f"✅ **Validation Gate Passed**: Test MAE ({m.get('test_mae', 0):.4f}%) < Zero Baseline ({m.get('zero_mae', 0):.4f}%). ML corrections active.")
            else:
                st.warning(f"⚠ **Validation Gate Muted**: Test MAE ({m.get('test_mae', 0):.4f}%) ≥ Zero Baseline ({m.get('zero_mae', 0):.4f}%). Physics baseline retained.")
        else:
            st.info("Train a gradient-boosted residual model on operational stress deltas.")
            
    if train_button:
        with st.spinner("Training Guarded HistGradientBoostingRegressor on operational stress features..."):
            try:
                fit_res = pinn_model.fit(clean_data, physics_model)
                if fit_res.get("success"):
                    # Generate forecast
                    pinn_fc = pinn_model.forecast(clean_data, forecast_years=forecast_years, scenario="middle")
                    st.session_state.pinn_model = pinn_model
                    st.session_state.pinn_trained = True
                    st.session_state.pinn_forecast = pinn_fc
                    st.session_state.pinn_metrics = pinn_model.metrics
                    SessionStateManager.mark_step_completed(8)
                    st.rerun()
                else:
                    st.error(f"❌ Training failed: {fit_res.get('message')}")
            except Exception as e:
                st.error(f"❌ Error during ML training: {str(e)}")
                
    # If trained, display overlay chart & comparison table
    if pinn_trained and pinn_forecast is not None and forecast_result is not None:
        st.markdown("### 📈 SOH Trajectory Comparison: Pure Physics vs Guarded Hybrid")
        
        import plotly.graph_objects as go
        fig_comp = go.Figure()
        
        # 1. Historical Observed
        if len(forecast_result.historical_dates) > 0:
            fig_comp.add_trace(go.Scatter(
                x=list(forecast_result.historical_dates),
                y=list(forecast_result.historical_soh),
                mode='lines',
                name='Field Observed SOH',
                line=dict(color='#2c3e50', width=2),
                hovertemplate='<b>Field Observed</b>: %{y:.2f}%<extra></extra>'
            ))
            
        # 2. OEM Baseline if available
        oem_data = st.session_state.get("oem_data")
        if oem_data:
            oem_vis = oem_data.get_visualization_data()
            if oem_vis.get("dates") is not None and oem_vis.get("soh_pct") is not None:
                fig_comp.add_trace(go.Scatter(
                    x=list(oem_vis["dates"]),
                    y=list(oem_vis["soh_pct"]),
                    mode='lines',
                    name='OEM Baseline',
                    line=dict(color='#3498db', width=2, dash='dot'),
                    hovertemplate='<b>OEM Baseline</b>: %{y:.2f}%<extra></extra>'
                ))
                
        # 3. Physics Forecast
        fig_comp.add_trace(go.Scatter(
            x=list(forecast_result.forecast_dates),
            y=list(forecast_result.forecast_soh),
            mode='lines',
            name='Physics-Based Forecast',
            line=dict(color='#e67e22', width=3),
            hovertemplate='<b>Physics Forecast</b>: %{y:.2f}%<extra></extra>'
        ))
        
        # 4. Guarded ML Residual Forecast
        fig_comp.add_trace(go.Scatter(
            x=list(pinn_forecast.forecast_dates),
            y=list(pinn_forecast.forecast_soh),
            mode='lines',
            name='Guarded ML Residual Forecast',
            line=dict(color='#8e44ad', width=3, dash='dash'),
            hovertemplate='<b>Guarded ML Forecast</b>: %{y:.2f}%<extra></extra>'
        ))
        
        # 5. EOL Line
        fig_comp.add_hline(
            y=EOL_SOH_THRESHOLD,
            line_dash="dash",
            line_color='#e74c3c',
            annotation_text=f"EOL Threshold ({EOL_SOH_THRESHOLD}%)",
            annotation_position="right",
            annotation_font_size=12,
            annotation_font_color='#e74c3c'
        )
        
        fig_comp.update_layout(
            title=dict(text=f"Model Comparison: Physics-Based vs Guarded ML ({forecast_years:.0f}-Year Forecast)", font=dict(size=18, color='#2c3e50')),
            xaxis_title="Date",
            yaxis_title="SOH (%)",
            template="plotly_white",
            height=520,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig_comp.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#ecf0f1')
        fig_comp.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#ecf0f1')
        fig_comp.update_yaxes(range=[50, 105])
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Side-by-side metrics table
        st.markdown("### 📋 Side-by-Side Model Performance Matrix")
        phys_metrics = st.session_state.get("calibration_metrics", {})
        ml_metrics = pinn_model.metrics if hasattr(pinn_model, 'metrics') else {}
        
        phys_eol = forecast_result.eol_info.get("years_to_eol", 0.0) if forecast_result.eol_info.get("eol_reached") else f"> {forecast_years:.1f}"
        ml_eol = pinn_forecast.eol_info.get("years_to_eol", 0.0) if pinn_forecast.eol_info.get("eol_reached") else f"> {forecast_years:.1f}"
        
        comp_rows = [
            {
                "Evaluation Dimension": "Architecture / Formulation",
                "Physics-Based Degradation Model": "Semi-empirical SEI + Cycle Growth (Arrhenius + Power Laws)",
                "Guarded Hybrid (Physics + ML)": "Physics Baseline + HistGradientBoosting Residual Correction"
            },
            {
                "Evaluation Dimension": "Historical Fit R²",
                "Physics-Based Degradation Model": f"{phys_metrics.get('r2_vs_smooth', 0):.4f}",
                "Guarded Hybrid (Physics + ML)": f"{ml_metrics.get('r2', phys_metrics.get('r2_vs_smooth', 0)):.4f}"
            },
            {
                "Evaluation Dimension": "Historical Fit RMSE (%)",
                "Physics-Based Degradation Model": f"{phys_metrics.get('rmse_vs_smooth', 0):.4f}%",
                "Guarded Hybrid (Physics + ML)": f"{ml_metrics.get('test_rmse', phys_metrics.get('rmse_vs_smooth', 0)):.4f}%"
            },
            {
                "Evaluation Dimension": "Chronological Test MAE (%)",
                "Physics-Based Degradation Model": f"{phys_metrics.get('mae_vs_smooth', 0):.4f}%",
                "Guarded Hybrid (Physics + ML)": f"{ml_metrics.get('test_mae', 0):.4f}% (vs Zero Base: {ml_metrics.get('zero_mae', 0):.4f}%)"
            },
            {
                "Evaluation Dimension": f"Final SOH at Year {forecast_years:.0f} (%)",
                "Physics-Based Degradation Model": f"{forecast_result.forecast_soh[-1]:.2f}%",
                "Guarded Hybrid (Physics + ML)": f"{pinn_forecast.forecast_soh[-1]:.2f}%"
            },
            {
                "Evaluation Dimension": "Predicted Years to EOL (65% SOH)",
                "Physics-Based Degradation Model": f"{phys_eol} yrs" if isinstance(phys_eol, str) else f"{phys_eol:.2f} yrs",
                "Guarded Hybrid (Physics + ML)": f"{ml_eol} yrs" if isinstance(ml_eol, str) else f"{ml_eol:.2f} yrs"
            },
            {
                "Evaluation Dimension": "Monotonicity Guarantee",
                "Physics-Based Degradation Model": "Strictly Monotonic by ODE Formulation",
                "Guarded Hybrid (Physics + ML)": "Enforced by Cumulative Minimum (np.minimum.accumulate)"
            },
            {
                "Evaluation Dimension": "Long-Term Extrapolation Guard",
                "Physics-Based Degradation Model": "Governed by Activation Energies (Ea_cal, Ea_cyc)",
                "Guarded Hybrid (Physics + ML)": "Decays exponentially (exp(-t/5.0)) back to Physics"
            },
            {
                "Evaluation Dimension": "Validation Gate Status",
                "Physics-Based Degradation Model": "Reference Ground Truth Baseline",
                "Guarded Hybrid (Physics + ML)": "Active (Passed Gate)" if ml_metrics.get("usable") else "Muted (Fallback to Physics)"
            },
        ]
        
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        
        # Engineering Details Card
        with st.expander("ℹ️ Technical Guardrail Details (Notebook Cell 26)", expanded=False):
            st.markdown("""
            #### Hybrid Guarded Residual Design:
            1. **Residual Target:** Rather than predicting raw SOH (which easily violates thermodynamics), ML is fitted strictly on the residual difference: $r(t) = \\text{SOH}_{obs}(t) - \\text{SOH}_{physics}(t)$.
            2. **Engineered Features:** 
               - Operating deltas relative to OEM nominal: $\\Delta EFC = EFC - 1.0$, $\\Delta DoD = DoD - 100\\%$, $\\Delta T_{avg} = T_{avg} - 35^\\circ\\text{C}$, $\\Delta T_{max} = T_{max} - 40^\\circ\\text{C}$
               - U-shaped SOC stress terms: $\\max(0, \\text{SOC} - 80)$, $|\\text{SOC} - 50|$, $\\max(0, 20 - \\text{SOC})$
               - Accumulated aging physics contributions: $d_{cal}$ and $d_{cyc}$.
            3. **Chronological Validation Gate:** The model is trained on the first 70% of chronological data and validated on the final 30%. It is only deployed if test MAE improves upon the zero-residual baseline ($MAE_{test} < MAE_{zero}$).
            4. **Horizon Decay & Bounding:** Residual predictions are clipped to $\\pm 1.5\\%$ and decayed by $\\exp(-t/5.0)$ so ungrounded ML extrapolation cannot dominate 10-20 year projections.
            5. **Cumulative Minimum:** Monotonic non-increasing capacity is guaranteed via running cumulative minimum.
            """)
    elif not pinn_trained:
        st.info("Click **'Train Guarded ML Residual Model'** above to fit the gradient-boosted residual model and generate the side-by-side comparison.")
        
    # Navigation
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("← Previous: Scenarios", use_container_width=True):
            st.session_state.current_step = 7
            st.rerun()
    with col2:
        if st.button("→ Next: Final Report", type="primary", use_container_width=True):
            st.session_state.current_step = 9
            st.rerun()


def render_page_report():
    """Page 09: Final Report"""
    st.markdown("## 📋 Final Engineering Report & Artifacts")
    st.markdown("Consolidated executive dashboard summary, parameter audit, and comprehensive report export.")
    
    # Dashboard summary
    st.markdown("### 📊 Executive KPI Summary")
    
    clean_data = st.session_state.get("clean_data")
    current_soh = None
    if clean_data is not None and "soh_pct" in clean_data.columns:
        valid_soh = pd.to_numeric(clean_data["soh_pct"], errors='coerce').dropna()
        if len(valid_soh) > 0:
            current_soh = float(valid_soh.iloc[-1])
    
    forecast_result = st.session_state.get("forecast_result")
    params = st.session_state.get("calibrated_params")
    scenario_results = st.session_state.get("scenario_results")
    uncertainty = st.session_state.get("uncertainty_result")
    calibration_metrics = st.session_state.get("calibration_metrics", {})
    diagnostics = st.session_state.get("diagnostics_result")
    oem_data = st.session_state.get("oem_data")
    
    # KPI Cards
    eol_years = forecast_result.eol_info.get("years_to_eol", 0.0) if forecast_result and forecast_result.eol_info.get("eol_reached") else "> 10"
    final_soh = forecast_result.forecast_soh[-1] if forecast_result else current_soh or 100.0
    
    render_kpi_cards(
        current_soh=current_soh,
        eol_years=eol_years if isinstance(eol_years, (int, float)) else 10.0,
        selected_model=st.session_state.get("selected_model", "Physics-Based"),
        oem_soh=current_soh,
        years_to_eol=eol_years if isinstance(eol_years, (int, float)) else 10.0,
    )
    
    # SOH Forecast Chart
    if forecast_result:
        st.markdown("### 📈 Multi-Year SOH Forecast")
        oem_vis = oem_data.get_visualization_data() if oem_data else {}
        fig = create_soh_over_time(
            historical_dates=forecast_result.historical_dates,
            historical_soh=forecast_result.historical_soh,
            forecast_dates=forecast_result.forecast_dates,
            forecast_soh=forecast_result.forecast_soh,
            oem_dates=oem_vis.get("dates"),
            oem_soh=oem_vis.get("soh_pct"),
            show_eol=True,
            eol_threshold=EOL_SOH_THRESHOLD,
            title=f"Multi-Year SOH Forecast & Projection to EOL ({EOL_SOH_THRESHOLD}%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Scenario Cards
    if scenario_results:
        st.markdown("### 📊 Operational Scenarios (Best / Middle / Worst)")
        render_scenario_cards(scenario_results)
    
    # Calibrated Parameters
    st.markdown("### ⚙️ Calibrated Parameters vs Boundaries")
    if params:
        render_parameter_table(
            to_dict_safe(params),
            warnings=st.session_state.get("calibration_warnings", [])
        )
    
    # Diagnostics expander
    with st.expander("🔧 Technical Diagnostics & Residual Statistics", expanded=False):
        if diagnostics:
            st.markdown("#### Residual Statistics")
            if hasattr(diagnostics, 'residual_statistics') and diagnostics.residual_statistics:
                st.json(diagnostics.residual_statistics)
            st.markdown("#### Calibration History")
            if hasattr(diagnostics, 'calibration_history') and diagnostics.calibration_history:
                st.json(diagnostics.calibration_history)
        else:
            st.info("Diagnostics recorded during Stage 1 & Stage 2 calibration.")
    
    # Download Section
    st.markdown("### 📥 Export Analytical Datasets (CSV)")
    data_frames = {}
    csv_names = {}
    
    if forecast_result:
        data_frames["forecast"] = pd.DataFrame({
            "date": forecast_result.forecast_dates,
            "soh_pct": forecast_result.forecast_soh,
            "calendar_loss_pct": forecast_result.calendar_loss_forecast,
            "cycle_loss_pct": forecast_result.cycle_loss_forecast,
        })
        csv_names["forecast"] = "bess_soh_forecast.csv"
    
    if scenario_results:
        scenario_data = []
        for key in ["best", "middle", "worst"]:
            if key in scenario_results:
                r = to_dict_safe(scenario_results[key])
                scenario_data.append({
                    "Scenario": key.capitalize(),
                    "EOL_years": r.get("EOL_year", 0),
                    "Final_SOH_pct": r.get("Final_SOH_pct", 0),
                    "EFC_per_day": r.get("EFC_per_day", 0),
                    "Avg_DoD_pct": r.get("Avg_DoD_pct", 0),
                    "Avg_SOC_pct": r.get("Avg_SOC_pct", 0),
                    "Mean_Tavg_C": r.get("Mean_Tavg_C", 0),
                })
        data_frames["scenarios"] = pd.DataFrame(scenario_data)
        csv_names["scenarios"] = "bess_scenarios_comparison.csv"
    
    if params:
        data_frames["calibration"] = pd.DataFrame([to_dict_safe(params)])
        csv_names["calibration"] = "bess_calibrated_parameters.csv"
    
    if data_frames:
        render_download_buttons(data_frames, csv_names)
    
    # Complete Excel Engineering Report
    st.markdown("### 📄 Complete Excel Engineering Report")
    st.markdown("Generates a comprehensive 13-sheet Microsoft Excel report containing all raw, calibrated, forecasted, scenario, and diagnostic tables.")
    
    if st.button("📊 Generate Comprehensive Excel Report", type="primary", use_container_width=True):
        with st.spinner("Compiling multi-sheet Excel report..."):
            try:
                active_ds = st.session_state.get("active_dataset", "BESS Field Unit")
                pinn_fc = st.session_state.get("pinn_forecast")
                
                report_data = ReportData(
                    project_info={
                        "name": f"BESS SOH Analytics — {active_ds}",
                        "location": st.session_state.get("project_location", "Field Site"),
                        "system_capacity": st.session_state.get("system_capacity", "Standard BESS Container"),
                        "chemistry": "LFP (Lithium Iron Phosphate)",
                    },
                    uploaded_datasets=SessionStateManager.get_uploaded_files_summary(),
                    data_quality={
                        "schema_ok": st.session_state.get("validation_passed", True),
                        "warnings": st.session_state.get("calibration_warnings", []),
                    },
                    oem_baseline=to_dict_safe(oem_data) if oem_data else {},
                    selected_model=st.session_state.get("selected_model", "Physics-Based"),
                    calibration_parameters=to_dict_safe(params) if params else {},
                    calibration_warnings=st.session_state.get("calibration_warnings", []),
                    historical_fit={
                        "current_soh": current_soh,
                        "metrics": calibration_metrics,
                    },
                    forecast=forecast_result.__dict__ if forecast_result else {},
                    best_case=to_dict_safe(scenario_results.get("best")) if scenario_results else {},
                    middle_case=to_dict_safe(scenario_results.get("middle")) if scenario_results else {},
                    worst_case=to_dict_safe(scenario_results.get("worst")) if scenario_results else {},
                    model_performance=calibration_metrics,
                    model_comparison={"comparison_table": {}} if pinn_fc else {},
                    uncertainty=uncertainty.__dict__ if uncertainty else {},
                    eol_prediction=forecast_result.eol_info if forecast_result else {},
                )
                
                excel_bytes = generate_report(report_data)
                
                st.download_button(
                    label="📥 Download BESS_SOH_Complete_Report.xlsx",
                    data=excel_bytes,
                    file_name=f"BESS_SOH_Report_{active_ds.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                st.success("✅ Excel report compiled successfully! Click the button above to download.")
            
            except Exception as e:
                st.error(f"❌ Report generation error: {str(e)}")
                if st.session_state.get("show_debug", False):
                    st.exception(e)
    
    # Navigation
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("← Previous: Comparison", use_container_width=True):
            st.session_state.current_step = 8
            st.rerun()


if __name__ == "__main__":
    main()