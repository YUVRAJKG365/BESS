import streamlit as st
from typing import Optional, Dict, Any
from utils.session_state import SessionStateManager
from utils.constants import EOL_SOH_THRESHOLD
from utils.formatting import format_number, format_percentage, format_years, kpi_card, status_badge


def render_kpi_cards(current_soh: Optional[float] = None, 
                      eol_years: Optional[float] = None,
                      selected_model: str = "Physics-Based",
                      oem_soh: Optional[float] = None,
                      years_to_eol: Optional[float] = None) -> None:
    """Render top KPI cards for the dashboard."""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        value = format_percentage(current_soh) if current_soh else "N/A"
        eol_target = SessionStateManager.get("eol_threshold", EOL_SOH_THRESHOLD)
        delta = f"Target: {format_percentage(eol_target)}"
        st.markdown(kpi_card("Current SOH", value, delta, delta_color="normal",
                            help_text="Last observed SOH"), unsafe_allow_html=True)
    
    with col2:
        value = format_years(eol_years)
        delta = f"Model: {selected_model}"
        st.markdown(kpi_card("Predicted EOL", value, delta, delta_color="warning",
                            help_text="Time until SOH reaches 65% threshold"), unsafe_allow_html=True)
    
    with col3:
        value = format_years(years_to_eol)
        delta = f"Current SOH: {format_percentage(current_soh)}" if current_soh else ""
        st.markdown(kpi_card("Years to EOL", value, delta, delta_color="normal",
                            help_text="Time remaining until end of life"), unsafe_allow_html=True)
    
    with col4:
        value = format_percentage(oem_soh) if oem_soh else "N/A"
        delta = "Reference baseline"
        st.markdown(kpi_card("OEM Baseline", value, delta, delta_color="info",
                            help_text="OEM baseline SOH"), unsafe_allow_html=True)
    
    with col5:
        value = selected_model.split("-")[0].strip() if "-" in selected_model else selected_model
        delta = f"Status: ✓ Active"
        st.markdown(kpi_card("Selected Model", value, delta, delta_color="info",
                            help_text="Currently active forecasting model"), unsafe_allow_html=True)


def render_file_upload_card(title: str, file_info: Dict[str, Any],
                            on_delete: Optional[str] = None) -> None:
    """Render a card for each uploaded file."""
    status = file_info.get("status", "valid")
    status_label = "✓ Valid" if status == "valid" else f"⚠ {status}"
    badge_html = status_badge(status, status_label)
    
    card_html = (
        f'<div style="background:white;border-radius:12px;padding:16px;margin:8px 0;border:1px solid #e1e8ed;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div>'
        f'<strong style="color:#2c3e50;">📄 {file_info.get("name", "Unknown")}</strong>'
        f'<div style="font-size:0.85rem;color:#7f8c8d;">'
        f'{file_info.get("type", "")} • {file_info.get("rows", 0):,} rows × {file_info.get("columns", 0)} cols • '
        f'{(file_info.get("size_mb", 0)):.2f} MB'
        f'</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'{badge_html}'
        f'<div style="font-size:0.85rem;color:#7f8c8d;margin-top:4px;">{file_info.get("date_range", "N/A")}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_upload_result_card(file_name: str, file_obj, metadata: Dict[str, Any]) -> None:
    """Render upload result with detailed information."""
    file_type = metadata.get("file_type", "unknown")
    rows = metadata.get("rows", 0)
    cols = metadata.get("columns", 0)
    date_range = metadata.get("date_range", (None, None))
    missing = metadata.get("missing_values", 0)
    duplicates = metadata.get("duplicate_rows", 0)
    
    # Date range display
    if date_range[0] and date_range[1]:
        date_str = f"{date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}"
    else:
        date_str = "N/A"
    
    status = "valid" if missing < rows * 0.3 and duplicates < rows * 0.1 else "warning"
    border_color = '#27ae60' if status == 'valid' else '#f39c12'
    badge_html = status_badge(status, '✓ Valid' if status == 'valid' else '⚠ Warning')
    missing_color = '#27ae60' if missing == 0 else '#f39c12'
    dup_color = '#27ae60' if duplicates == 0 else '#f39c12'

    card_html = (
        f'<div style="background:#f8f9fa;border-radius:12px;padding:16px;margin:8px 0;border-left:4px solid {border_color};">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div>'
        f'<strong style="color:#2c3e50;">📄 {file_name}</strong>'
        f'<div style="font-size:0.85rem;color:#7f8c8d;margin-top:2px;">{file_type.upper()} • {rows:,} rows × {cols} columns</div>'
        f'</div>'
        f'<div>{badge_html}</div>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:12px;margin-top:12px;">'
        f'<div style="text-align:center;padding:8px;background:white;border-radius:8px;">'
        f'<div style="font-size:0.7rem;color:#95a5a6;">Date Range</div>'
        f'<div style="font-size:0.8rem;font-weight:600;color:#2c3e50;">{date_str}</div>'
        f'</div>'
        f'<div style="text-align:center;padding:8px;background:white;border-radius:8px;">'
        f'<div style="font-size:0.7rem;color:#95a5a6;">Missing</div>'
        f'<div style="font-size:0.8rem;font-weight:600;color:{missing_color};">{missing:,}</div>'
        f'</div>'
        f'<div style="text-align:center;padding:8px;background:white;border-radius:8px;">'
        f'<div style="font-size:0.7rem;color:#95a5a6;">Duplicates</div>'
        f'<div style="font-size:0.8rem;font-weight:600;color:{dup_color};">{duplicates:,}</div>'
        f'</div>'
        f'<div style="text-align:center;padding:8px;background:white;border-radius:8px;">'
        f'<div style="font-size:0.7rem;color:#95a5a6;">Variables</div>'
        f'<div style="font-size:0.8rem;font-weight:600;color:#2c3e50;">{cols}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_validation_badge(is_valid: bool, label: str) -> str:
    """Render a validation badge."""
    if is_valid:
        return f'<span style="color: #27ae60; font-weight: 600;">✓ {label}</span>'
    else:
        return f'<span style="color: #e74c3c; font-weight: 600;">✕ {label}</span>'


def render_warning_card(message: str, category: str = "warning") -> None:
    """Render a warning card."""
    colors = {
        "warning": {"bg": "#fef9e7", "border": "#f39c12", "icon": "⚠"},
        "error": {"bg": "#fdf2f2", "border": "#e74c3c", "icon": "✕"},
        "info": {"bg": "#eaf2f8", "border": "#3498db", "icon": "ℹ"},
    }
    
    c = colors.get(category, colors["warning"])
    
    st.markdown(f"""
    <div style="background: {c['bg']}; border-left: 4px solid {c['border']}; 
                border-radius: 8px; padding: 12px 16px; margin: 8px 0;">
        <span style="font-size: 1.2rem;">{c['icon']}</span>
        <span style="color: #2c3e50; font-size: 0.9rem; margin-left: 8px;">{message}</span>
    </div>
    """, unsafe_allow_html=True)