import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from utils.session_state import SessionStateManager
from utils.formatting import format_number, format_percentage, format_date


def render_data_table(df: pd.DataFrame, max_rows: int = 100, 
                         title: Optional[str] = None,
                         show_index: bool = False) -> None:
    """Render a data table with proper formatting."""
    if title:
        st.subheader(title)
    
    display_df = df.copy()
    
    # Format numeric columns
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            # Don't round if it's datetime-like
            continue
    
    # Show data table
    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(max_rows * 30, 400),
    )


def render_metadata_table(metadata_list: List[Dict[str, Any]]) -> None:
    """Render metadata table for uploaded files."""
    if not metadata_list:
        st.info("No files uploaded yet.")
        return
    
    rows = []
    for meta in metadata_list:
        date_start = meta.get("date_range", (None, None))[0]
        date_end = meta.get("date_range", (None, None))[1]
        date_str = f"{format_date(date_start)} to {format_date(date_end)}" if date_start else "N/A"
        
        rows.append({
            "File": meta.get("name", ""),
            "Type": meta.get("file_type", "").upper(),
            "Rows": meta.get("rows", 0),
            "Columns": meta.get("columns", 0),
            "Missing": meta.get("missing_values", 0),
            "Duplicates": meta.get("duplicate_rows", 0),
            "Date Range": date_str,
            "SOH Column": meta.get("detected_soh_column", ""),
            "Temperature": ", ".join(meta.get("detected_temperature_columns", [])),
            "SOC": meta.get("detected_soc_column", ""),
            "DoD": meta.get("detected_dod_column", ""),
            "EFC": meta.get("detected_efc_column", ""),
            "Status": "✓ Valid" if meta.get("validation_status") == "valid" else "⚠ Check",
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    
    # Export button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Metadata Summary",
        data=csv,
        file_name="dataset_metadata.csv",
        mime="text/csv",
    )


def render_column_mapping_table(mappings: Dict[str, str], 
                                   available_columns: List[str],
                                   on_change: Optional[callable] = None) -> None:
    """Render interactive column mapping table."""
    st.subheader("📋 Column Mapping")
    
    # Get model variables and their defaults
    model_vars = {
        "soh_pct": "SOH (%)",
        "temperature": "Temperature (°C)",
        "SOC_pct": "SOC (%)",
        "Avg_DoD_pct": "Average DoD (%)",
        "EFC_per_day": "EFC/day",
        "Cycles_per_day": "Cycles/day",
        "Charge_Duration_hr": "Charge Duration (hr)",
        "Discharge_Duration_hr": "Discharge Duration (hr)",
        "C_rate": "C-rate",
        "time": "Time",
        "Tmax": "Max Temperature (°C)",
        "Tmin": "Min Temperature (°C)",
        "power_kw": "Power (kW)",
        "voltage_v": "Voltage (V)",
        "current_a": "Current (A)",
    }
    
    rows = []
    for model_var, display_name in model_vars.items():
        current_mapping = mappings.get(model_var, "")
        if current_mapping and current_mapping in available_columns:
            # This mapping is set to an existing column
            pass
        
        rows.append({
            "Model Variable": model_var,
            "Display Name": display_name,
            "Uploaded Column": current_mapping if current_mapping else "(not mapped)",
            "Status": "✓ Mapped" if current_mapping else "⚠ Unmapped",
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)
    
    # Allow user to change mappings
    st.markdown("### Modify Column Mapping")
    for model_var in model_vars:
        cols = [None] + available_columns
        col_labels = ["(none/unmapped)"] + available_columns
        
        current = mappings.get(model_var, "")
        try:
            idx = cols.index(current) if current in cols else 0
        except ValueError:
            idx = 0
        
        new_val = st.selectbox(
            f"**{model_var}** →",
            cols,
            index=idx,
            key=f"mapping_{model_var}",
        )
        
        if new_val != current and new_val:
            mappings[model_var] = new_val
            st.session_state.column_mappings = mappings
    
    # Save button
    if st.button("💾 Save Mappings", use_container_width=True):
        st.success("Column mappings saved!")
        st.rerun()


def render_validation_summary(validation_results: Dict[str, Any]) -> None:
    """Render validation summary with status indicators."""
    st.subheader("📋 Validation Summary")
    
    categories = [
        ("File Validation", "file_ok", "✓ File structure valid"),
        ("Schema Validation", "schema_ok", "✓ All required columns present"),
        ("Date/Time Validation", "date_ok", "✓ Dates parsed correctly"),
        ("Numeric Validation", "numeric_ok", "✓ Numeric values valid"),
        ("Missing Values", "missing_ok", "✓ Acceptable missing values"),
        ("Duplicate Rows", "duplicate_ok", "✓ Duplicate count acceptable"),
        ("Physical Plausibility", "physical_ok", "✓ Values within physical range"),
        ("Model Compatibility", "model_compat_ok", "✓ Compatible with model"),
    ]
    
    for label, key, message in categories:
        is_ok = validation_results.get(key, False)
        status = "✓" if is_ok else "⚠"
        color = "#27ae60" if is_ok else "#f39c12"
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; padding: 4px 0;">
            <span style="color: {color}; font-weight: bold;">{status}</span>
            <span style="margin-left: 8px; color: #2c3e50;">{message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Show warnings and errors
    warnings = validation_results.get("warnings", [])
    errors = validation_results.get("errors", [])
    
    if warnings:
        st.markdown("### ⚠ Warnings")
        for w in warnings:
            st.warning(w)
    
    if errors:
        st.markdown("### ✕ Errors")
        for e in errors:
            st.error(e)
    
    # Overall status
    is_valid = validation_results.get("valid", False)
    if is_valid:
        st.success("✅ All validations passed! Ready for forecasting.")
    else:
        st.error("❌ Validation failed. Please fix issues before continuing.")


def render_step_progress(current_step: int, total_steps: int = 10) -> None:
    """Render step progress bar."""
    progress = (current_step + 1) / total_steps
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50;">
                Step {current_step + 1} of {total_steps}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.progress(progress, text=f"{int(progress * 100)}% complete")


def render_upload_progress(files: List, status: str = "uploading") -> None:
    """Render upload progress."""
    if status == "uploading":
        st.info(f"📤 Uploading {len(files)} file(s)...")
        progress = min(len(files) / 10, 1.0)
        st.progress(progress, text=f"{len(files)} files received")
    
    elif status == "processing":
        st.info("⚙️ Processing files...")
        st.progress(0.5, text="Parsing and validating...")
    
    elif status == "complete":
        st.success(f"✅ {len(files)} files uploaded successfully!")


def render_parameter_table(params: Dict[str, float], 
                             warnings: Optional[List[str]] = None) -> None:
    """Render calibration parameter table."""
    st.subheader("⚙️ Calibration Parameters")
    
    from utils.constants import PARAM_BOUNDS
    
    param_meanings = {
        "A_cal": "Calendar aging pre-exponential factor",
        "A_cyc": "Cycle aging pre-exponential factor",
        "Ea_cal": "Calendar activation energy (J/mol)",
        "Ea_cyc": "Cycle activation energy (J/mol)",
        "alpha": "Calendar aging time exponent",
        "beta": "Cycle aging throughput exponent",
        "k_SOC_high": "High SOC stress factor (>80%)",
        "k_SOC_reward": "Mid SOC reward factor (20-80%)",
        "k_SOC_low": "Low SOC stress factor (<20%)",
        "gamma_DoD": "DoD stress exponent",
        "k_C": "C-rate stress factor",
        "k_Tmax": "High temperature (Tmax) stress factor",
        "k_Tmin": "Low temperature (Tmin) stress factor",
    }
    
    rows = []
    for name, value in params.items():
        bounds = PARAM_BOUNDS.get(name, (None, None))
        
        # Check if at bound
        status = ""
        if bounds[0] is not None and abs(value - bounds[0]) < 1e-6:
            status = "⚠ AT LOWER BOUND"
        elif bounds[1] is not None and abs(value - bounds[1]) < 1e-6:
            status = "⚠ AT UPPER BOUND"
        else:
            status = "✓ OK"
        
        rows.append({
            "Parameter": name,
            "Value": value,
            "Lower Bound": bounds[0] if bounds[0] else "–",
            "Upper Bound": bounds[1] if bounds[1] else "–",
            "Status": status,
            "Meaning": param_meanings.get(name, ""),
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    
    # Show warnings for bound issues
    if warnings:
        st.markdown("### ⚠ Calibration Warnings")
        for w in warnings:
            if "LOWER" in w or "UPPER" in w:
                st.warning(f"⚠ {w}: Parameter at bound - calibration may be under-constrained")


def render_comparison_table(metrics: Dict[str, Dict], 
                             model_names: List[str] = None) -> None:
    """Render model comparison table."""
    if model_names is None:
        model_names = list(metrics.keys())
    
    # Get all metric names
    all_metrics = set()
    for m in metrics.values():
        all_metrics.update(m.keys())
    
    rows = []
    for metric in sorted(all_metrics):
        row = {"Metric": metric}
        for name in model_names:
            val = metrics[name].get(metric, "")
            if isinstance(val, float):
                row[name] = f"{val:.4f}"
            else:
                row[name] = str(val) if val else ""
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)


def render_download_buttons(
    data_frames: Dict[str, pd.DataFrame],
    csv_names: Dict[str, str] = None
) -> None:
    """Render download buttons for dataframes."""
    st.markdown("### 📥 Download Results")
    
    cols = st.columns(len(data_frames))
    
    for i, (name, df) in enumerate(data_frames.items()):
        with cols[i]:
            csv = df.to_csv(index=False)
            filename = csv_names.get(name, f"{name}.csv")
            
            st.download_button(
                label=f"📥 {name}",
                data=csv,
                file_name=filename,
                mime="text/csv",
                use_container_width=True,
            )