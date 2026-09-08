import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from utils.session_state import SessionStateManager
from utils.formatting import status_badge
from components.cards import render_upload_result_card, kpi_card
from components.tables import render_metadata_table
from utils.helpers import detect_file_type


class FileUploader:
    """Handles multi-file upload with metadata extraction."""
    
    def __init__(self):
        self.uploaded_files: List = []
        self.metadata: Dict[str, Dict] = {}
    
    def render_upload_section(self) -> None:
        """Render the main file upload interface."""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 24px;">
            <h2 style="color: #2c3e50;">📁 Data Upload</h2>
            <p style="color: #7f8c8d; font-size: 1.1rem;">
                Upload multiple BESS field data files (Excel/CSV) for degradation analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload widget
        st.markdown("### Upload Field Data Files")
        
        uploaded = st.file_uploader(
            "Choose BESS data files",
            type=['xlsx', 'xls', 'csv'],
            accept_multiple_files=True,
            help="Upload one or more field data files. Each file should contain SOH, temperature, SOC, and other operating data.",
            key="field_data_uploader",
        )
        
        if uploaded:
            for file_obj in uploaded:
                self._process_uploaded_file(file_obj)
            
            # Show summary
            self._render_upload_summary()
    
    def _process_uploaded_file(self, file_obj) -> None:
        """Process a single uploaded file."""
        file_name = file_obj.name
        file_type = detect_file_type(file_name)
        
        try:
            # Load data
            import pandas as pd
            if file_type == 'excel':
                xls = pd.ExcelFile(file_obj)
                df = pd.read_excel(file_obj, sheet_name=xls.sheet_names[0])
                sheet_name = xls.sheet_names[0]
            else:
                df = pd.read_csv(file_obj)
                sheet_name = "CSV"
            
            # Extract metadata
            metadata = self._extract_metadata(df, file_name, file_type, sheet_name)
            
            # Store raw dataframe in metadata for later use
            metadata["raw_df"] = df
            
            # Store
            self.uploaded_files.append(file_obj)
            self.metadata[file_name] = metadata
            
            # Add to session state
            SessionStateManager.add_file(file_obj, metadata)
            
            # Auto-detect column mappings and compute clean_data
            self._compute_clean_data_from_uploads()
            
            # Mark upload step complete
            SessionStateManager.mark_step_completed(0)
            
        except Exception as e:
            st.error(f"❌ Error processing {file_name}: {str(e)}")
            if st.session_state.get("show_debug", False):
                st.exception(e)
    
    def _compute_clean_data_from_uploads(self) -> None:
        """Compute clean_data from all uploaded files with auto-detected mappings."""
        from backend.data_mapper import DataMapper, COLUMN_MAPPINGS
        
        files = SessionStateManager.get_uploaded_files_summary()
        if not files:
            return
        
        mapper = DataMapper()
        metadata_dict = SessionStateManager.get("file_metadata") or {}
        
        # Get first file's columns to auto-detect mappings
        first_file_name = files[0]["name"]
        first_metadata = metadata_dict.get(first_file_name, {})
        raw_df = first_metadata.get("raw_df")
        
        if raw_df is None:
            return
        
        # Auto-detect column mappings
        mapped = mapper.auto_detect_mappings(raw_df)
        st.session_state.column_mappings = mapped
        
        # Process all files
        all_dfs = []
        for file_info in files:
            file_name = file_info["name"]
            meta = metadata_dict.get(file_name, {})
            raw_df = meta.get("raw_df")
            
            if raw_df is not None:
                cleaned = mapper.clean_data(raw_df, mapped)
                all_dfs.append(cleaned)
        
        if all_dfs:
            clean_data = pd.concat(all_dfs, ignore_index=True)
            SessionStateManager.set_clean_data(clean_data)
            st.session_state.upload_complete = True
    
    def _extract_metadata(self, df: Any, file_name: str, 
                              file_type: str, sheet_name: str) -> Dict[str, Any]:
        """Extract metadata from uploaded dataframe."""
        import pandas as pd
        import numpy as np
        from utils.helpers import calculate_dates_range, safe_float
        
        metadata = {
            "file_name": file_name,
            "file_type": file_type,
            "sheet_name": sheet_name,
            "rows": len(df),
            "columns": len(df.columns),
            "date_range": (None, None),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "variables": list(df.columns),
            "detected_soh_column": None,
            "detected_temperature_columns": [],
            "detected_soc_column": None,
            "detected_dod_column": None,
            "detected_cycles_column": None,
            "detected_efc_column": None,
            "detected_charge_duration": None,
            "detected_discharge_duration": None,
        }
        
        # Date range
        date_cols = []
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() > len(df) * 0.5:
                date_cols.append(col)
        
        if date_cols:
            date_range = calculate_dates_range(df, date_cols[0])
            metadata["date_range"] = date_range
        
        # Detect columns using DataMapper auto detection
        from backend.data_mapper import DataMapper
        auto_map = DataMapper().auto_detect_mappings(df)
        metadata["detected_soh_column"] = auto_map.get("soh_pct")
        metadata["detected_temperature_columns"] = [c for c in [auto_map.get("temperature"), auto_map.get("Tmax"), auto_map.get("Tmin")] if c]
        metadata["detected_soc_column"] = auto_map.get("SOC_pct")
        metadata["detected_dod_column"] = auto_map.get("Avg_DoD_pct")
        metadata["detected_cycles_column"] = auto_map.get("Cycles_per_day")
        metadata["detected_efc_column"] = auto_map.get("EFC_per_day")
        metadata["detected_charge_duration"] = auto_map.get("Charge_Duration_hr")
        metadata["detected_discharge_duration"] = auto_map.get("Discharge_Duration_hr")
        
        return metadata
    
    def _render_upload_summary(self) -> None:
        """Render summary of uploaded files."""
        st.markdown("### 📊 Uploaded Files Summary")
        
        files = SessionStateManager.get_uploaded_files_summary()
        
        if not files:
            return
        
        # Display each file as a card
        for file_info in files:
            metadata = self.metadata.get(file_info['name'], {})
            render_upload_result_card(
                file_info['name'], 
                None,  # file_obj
                metadata
            )
        
        # Consolidated view
        st.markdown("### 📋 Consolidated View")
        render_metadata_table(files)
        
        # Store in session state
        st.session_state.upload_complete = True
    
    def add_oem_file(self, file_obj) -> bool:
        """Add an OEM baseline file."""
        try:
            import pandas as pd
            file_name = file_obj.name
            file_type = detect_file_type(file_name)
            
            if file_type == 'excel':
                xls = pd.ExcelFile(file_obj)
                df = pd.read_excel(file_obj, sheet_name=xls.sheet_names[0])
            else:
                df = pd.read_csv(file_obj)
            
            # Store OEM data
            st.session_state.oem_raw_df = df
            st.session_state.oem_uploaded = True
            
            return True
        except Exception:
            return False


def render_oem_upload_section() -> None:
    """Render the OEM baseline upload section."""
    st.markdown("### 🏭 OEM Baseline Upload")
    
    st.markdown("""
    <p style="color: #7f8c8d;">
        Upload the OEM baseline file to provide a reference degradation trajectory. 
        This file should contain historical SOH data from the manufacturer or a reference system.
    </p>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Upload OEM Baseline File",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=False,
        help="OEM baseline SOH data",
        key="oem_uploader",
    )
    
    if uploaded:
        try:
            import pandas as pd
            from backend.oem_processor import OEMProcessor
            
            file_name = uploaded[0].name if isinstance(uploaded, list) else uploaded.name
            file_type = detect_file_type(file_name)
            
            if file_type == 'excel':
                xls = pd.ExcelFile(uploaded)
                sheet_to_use = "OEM_Baseline" if "OEM_Baseline" in xls.sheet_names else xls.sheet_names[0]
                df = pd.read_excel(uploaded, sheet_name=sheet_to_use)
            else:
                df = pd.read_csv(uploaded)
            
            processor = OEMProcessor()
            time_col = processor._detect_time_column(df)
            soh_col = processor._detect_soh_column(df)
            
            success = processor.load_oem_file(df, time_col=time_col, soh_col=soh_col)
            
            if success:
                st.session_state.oem_data = processor.oem_data
                st.session_state.oem_uploaded = True
                st.success(f"✅ OEM baseline loaded from {file_name}")
                
                # Show OEM data preview
                st.markdown("#### OEM Data Preview")
                st.dataframe(df.head(20), use_container_width=True)
                
                # Show OEM metadata
                st.markdown("#### OEM Baseline Statistics")
                metadata = processor.oem_data.metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Data Points", metadata.get("n_points", 0))
                with col2:
                    st.metric("Time Span", f"{metadata.get('time_span_years', 0):.1f} years")
                with col3:
                    st.metric("Initial SOH", f"{metadata.get('initial_soh', 0):.1f}%")
                with col4:
                    st.metric("Degradation Rate", f"{metadata.get('degradation_rate_pct_per_year', 0):.3f}%/yr")
                
                # Validation
                validation = processor.validate()
                if validation.get("valid"):
                    st.success("✅ OEM data validation passed")
                if validation.get("warnings"):
                    for w in validation["warnings"]:
                        st.warning(w)
                if validation.get("errors"):
                    for e in validation["errors"]:
                        st.error(e)
                
                # OEM visualization
                vis_data = processor.get_visualization_data()
                if vis_data.get("dates") is not None:
                    st.markdown("#### OEM Baseline SOH Curve")
                    
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(vis_data['dates']),
                        y=list(vis_data['soh_pct']),
                        mode='lines+markers',
                        name='OEM SOH',
                        line=dict(color='#3498db', width=2),
                        marker=dict(size=5, color='#3498db'),
                    ))
                    fig.update_layout(
                        title="OEM Baseline SOH",
                        xaxis_title="Date",
                        yaxis_title="SOH (%)",
                        template="plotly_white",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Failed to load OEM baseline. Check file format and columns.")
        
        except Exception as e:
            st.error(f"❌ Error uploading OEM file: {str(e)}")
            if st.session_state.get("show_debug", False):
                st.exception(e)
    
    # If OEM already uploaded, show status
    if st.session_state.get("oem_uploaded"):
        st.markdown("### OEM Status")
        st.success("✅ OEM baseline loaded and validated")
        
        # Option to upload a different file
        if st.button("🔄 Replace OEM Baseline", use_container_width=True):
            st.session_state.oem_uploaded = False
            st.rerun()