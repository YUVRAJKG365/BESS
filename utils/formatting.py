import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union
from datetime import datetime


def format_number(value: Any, decimals: int = 2, unit: str = "") -> str:
    """Format a number for display."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "N/A"
    
    try:
        if isinstance(value, (int, np.integer)):
            formatted = f"{int(value):,}"
        else:
            formatted = f"{float(value):,.{decimals}f}"
        
        if unit:
            formatted += f" {unit}"
        return formatted
    except Exception:
        return str(value)


def format_percentage(value: Any, decimals: int = 1) -> str:
    """Format as percentage."""
    return format_number(value, decimals=decimals, unit="%")


def format_years(value: Any, decimals: int = 1) -> str:
    """Format years."""
    if value is None or np.isinf(value):
        return "> 20 years" if value == np.inf else "N/A"
    return format_number(value, decimals=decimals, unit="years")


def format_date(value: Any) -> str:
    """Format date for display."""
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float)):
        try:
            return pd.Timestamp(value).strftime("%Y-%m-%d")
        except Exception:
            return str(value)
    return str(value)


def format_datetime(value: Any) -> str:
    """Format datetime for display."""
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def format_scientific(value: Any, decimals: int = 2) -> str:
    """Format in scientific notation."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "N/A"
    try:
        return f"{float(value):.{decimals}e}"
    except Exception:
        return str(value)


def status_badge(status: str, label: Optional[str] = None) -> str:
    """Generate HTML for status badge."""
    colors = {
        "valid": "#27ae60",
        "ok": "#27ae60",
        "success": "#27ae60",
        "warning": "#f39c12",
        "caution": "#f39c12",
        "error": "#e74c3c",
        "fail": "#e74c3c",
        "info": "#3498db",
        "unknown": "#95a5a6",
    }
    color = colors.get(status.lower(), "#95a5a6")
    display_label = label or status.capitalize()
    return f'<span style="display:inline-block;padding:4px 12px;border-radius:20px;background-color:{color};color:white;font-weight:600;font-size:0.85rem;">{display_label}</span>'


def kpi_card(title: str, value: str, delta: Optional[str] = None, 
             delta_color: str = "normal", help_text: Optional[str] = None) -> str:
    """Generate HTML for a KPI card."""
    delta_html = ""
    if delta:
        delta_color_map = {
            "normal": "#3498db",
            "inverse": "#e74c3c",
            "off": "#95a5a6",
        }
        color = delta_color_map.get(delta_color, "#3498db")
        delta_html = f'<div style="color:{color};font-size:0.85rem;margin-top:4px;">{delta}</div>'
    
    help_html = ""
    if help_text:
        help_html = f'<div style="font-size:0.75rem;color:#95a5a6;margin-top:4px;">{help_text}</div>'
    
    return (
        f'<div style="background:white;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);border:1px solid #e1e8ed;">'
        f'<div style="font-size:0.85rem;color:#7f8c8d;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;">{title}</div>'
        f'<div style="font-size:2rem;font-weight:700;color:#2c3e50;margin:8px 0;">{value}</div>'
        f'{delta_html}{help_html}'
        f'</div>'
    )


def scenario_card(scenario: Dict[str, Any], scenario_type: str) -> str:
    """Generate HTML for a scenario card."""
    type_colors = {
        "best": {"bg": "#e8f8f5", "border": "#27ae60", "icon": "✓"},
        "middle": {"bg": "#fef9e7", "border": "#f39c12", "icon": "◐"},
        "worst": {"bg": "#fdf2f2", "border": "#e74c3c", "icon": "✗"},
    }
    
    colors = type_colors.get(scenario_type.lower(), type_colors["middle"])
    
    eol_years = scenario.get("EOL_year", scenario.get("EOL (years)", "N/A"))
    final_soh = scenario.get("Final_SOH_pct", scenario.get("Final SOH (%)", "N/A"))
    efc = scenario.get("EFC_per_day", scenario.get("EFC/day", "N/A"))
    dod = scenario.get("Avg_DoD_pct", scenario.get("Avg DoD (%)", "N/A"))
    soc = scenario.get("Avg_SOC_pct", scenario.get("Avg SOC (%)", "N/A"))
    temp = scenario.get("Mean_Tavg_C", scenario.get("Mean Temp (°C)", "N/A"))
    tmax = scenario.get("Max_Tmax_C", scenario.get("Max Temp (°C)", "N/A"))
    desc = scenario.get("description", "")
    sc_name = scenario.get("scenario_name", scenario_type.capitalize())
    
    return (
        f'<div style="background:{colors["bg"]};border:2px solid {colors["border"]};border-radius:16px;padding:24px;margin:8px 0;">'
        f'<div style="display:flex;align-items:center;margin-bottom:16px;">'
        f'<span style="background:{colors["border"]};color:white;width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;margin-right:12px;">{colors["icon"]}</span>'
        f'<h3 style="margin:0;color:#2c3e50;font-size:1.25rem;">{sc_name} CASE</h3>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:repeat(2, 1fr);gap:12px;">'
        f'<div><div style="font-size:0.75rem;color:#7f8c8d;text-transform:uppercase;">Predicted EOL</div><div style="font-size:1.5rem;font-weight:700;color:{colors["border"]};">{format_years(eol_years)}</div></div>'
        f'<div><div style="font-size:0.75rem;color:#7f8c8d;text-transform:uppercase;">Final SOH</div><div style="font-size:1.5rem;font-weight:700;color:#2c3e50;">{format_percentage(final_soh)}</div></div>'
        f'<div><div style="font-size:0.75rem;color:#7f8c8d;text-transform:uppercase;">Avg EFC/day</div><div style="font-size:1.25rem;font-weight:600;color:#2c3e50;">{format_number(efc, 2)}</div></div>'
        f'<div><div style="font-size:0.75rem;color:#7f8c8d;text-transform:uppercase;">Avg DoD</div><div style="font-size:1.25rem;font-weight:600;color:#2c3e50;">{format_percentage(dod)}</div></div>'
        f'<div><div style="font-size:0.75rem;color:#7f8c8d;text-transform:uppercase;">Avg SOC</div><div style="font-size:1.25rem;font-weight:600;color:#2c3e50;">{format_percentage(soc)}</div></div>'
        f'<div><div style="font-size:0.75rem;color:#7f8c8d;text-transform:uppercase;">Mean Temp</div><div style="font-size:1.25rem;font-weight:600;color:#2c3e50;">{format_number(temp, 1)}°C</div></div>'
        f'<div><div style="font-size:0.75rem;color:#7f8c8d;text-transform:uppercase;">Max Temp</div><div style="font-size:1.25rem;font-weight:600;color:#2c3e50;">{format_number(tmax, 1)}°C</div></div>'
        f'<div><div style="font-size:0.75rem;color:#7f8c8d;text-transform:uppercase;">Cycle Intensity</div><div style="font-size:1.25rem;font-weight:600;color:#2c3e50;">{format_number(scenario.get("Cycles_per_day", efc), 2)}/day</div></div>'
        f'</div>'
        f'<div style="margin-top:16px;padding-top:16px;border-top:1px solid {colors["border"]}33;">'
        f'<div style="font-size:0.85rem;color:#7f8c8d;">{desc}</div>'
        f'</div>'
        f'</div>'
    )


def metric_row(label: str, value: str, unit: str = "") -> str:
    """Generate a metric row for tables."""
    return f'<tr><td style="padding:8px 12px;color:#7f8c8d;font-weight:500;">{label}</td><td style="padding:8px 12px;color:#2c3e50;font-weight:600;text-align:right;">{value}{" " + unit if unit else ""}</td></tr>'


def progress_bar(progress: float, label: str = "", color: str = "#3498db") -> str:
    """Generate HTML progress bar."""
    pct = max(0, min(100, progress * 100))
    return (
        f'<div style="margin:8px 0;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
        f'<span style="font-size:0.85rem;color:#2c3e50;">{label}</span>'
        f'<span style="font-size:0.85rem;color:#7f8c8d;">{pct:.0f}%</span>'
        f'</div>'
        f'<div style="height:8px;background:#ecf0f1;border-radius:4px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:4px;transition:width 0.3s ease;"></div>'
        f'</div>'
        f'</div>'
    )