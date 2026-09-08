import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from utils.session_state import SessionStateManager
from utils.formatting import scenario_card, format_number, format_percentage, format_years


def render_scenario_cards(scenario_results: Optional[Dict[str, Any]] = None) -> None:
    """Render the three main scenario cards (Best/Middle/Worst)."""
    
    if scenario_results is None or not scenario_results:
        st.info("No scenario results available. Run calibration and forecasting first.")
        return
    
    st.markdown("## 📊 Best / Middle / Worst Case Scenarios")
    st.markdown("""
    <p style="color: #7f8c8d;">
        Scenarios are derived from observed field data distributions.
        <b>Best Case</b> represents favorable operating conditions (lower stress),
        <b>Middle Case</b> represents median field conditions,
        <b>Worst Case</b> represents challenging operating conditions (higher stress).
    </p>
    """, unsafe_allow_html=True)
    
    # Scenario order
    scenario_order = ["best", "middle", "worst"]
    labels = {
        "best": "BEST CASE",
        "middle": "MIDDLE CASE",
        "worst": "WORST CASE",
    }
    
    # Render each scenario card
    for scenario_key in scenario_order:
        result = scenario_results.get(scenario_key)
        if result is None:
            continue
        
        # Convert to dict if needed
        if not isinstance(result, dict):
            result = result.to_dict() if hasattr(result, 'to_dict') else {}
        
        # Add label and description
        result["scenario_name"] = labels.get(scenario_key, scenario_key.capitalize())
        result["description"] = {
            "best": "Lower degradation stress from favorable operating conditions",
            "middle": "Representative/median field operating conditions",
            "worst": "Higher degradation stress from challenging operating conditions",
        }.get(scenario_key, "")
        
        # Render card
        st.markdown(scenario_card(result, scenario_key), unsafe_allow_html=True)
    
    # Comparison table
    st.markdown("### 📋 Scenario Comparison Table")
    
    rows = []
    for key in scenario_order:
        result = scenario_results.get(key)
        if result is not None:
            if not isinstance(result, dict):
                result = result.to_dict() if hasattr(result, 'to_dict') else {}
            if result:
                rows.append({
                    "Scenario": result.get("scenario_name", labels.get(key, key.capitalize())),
                    "EFC/day": f"{result.get('EFC_per_day', 0):.2f}",
                    "Avg SOC (%)": f"{result.get('Avg_SOC_pct', 0):.1f}",
                    "Avg DoD (%)": f"{result.get('Avg_DoD_pct', 0):.1f}",
                    "Mean Temp (°C)": f"{result.get('Mean_Tavg_C', 0):.1f}",
                    "Max Temp (°C)": f"{result.get('Max_Tmax_C', 0):.1f}",
                    "Cycles/day": f"{result.get('Cycles_per_day', 0):.2f}",
                    "C-rate": f"{result.get('C_rate', 0):.2f}",
                    "EOL (years)": f"{result.get('EOL_year', 0):.1f}",
                    "Final SOH (%)": f"{result.get('Final_SOH_pct', 0):.1f}",
                })
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # EOL comparison bar chart
    if rows:
        st.markdown("### 📈 EOL Comparison")
        
        eol_values = {}
        for row in rows:
            name = row["Scenario"].split(" ")[0]
            eol_values[row["Scenario"]] = float(row["EOL (years)"])
        
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Bar(
                x=list(eol_values.keys()),
                y=list(eol_values.values()),
                marker_color=['#27ae60', '#f39c12', '#e74c3c'],
                text=[f"{v:.1f} years" for v in eol_values.values()],
                textposition='outside',
            )
        ])
        fig.update_layout(
            title="Predicted EOL by Scenario",
            xaxis_title="Scenario",
            yaxis_title="EOL (years)",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_scenario_table(scenario_results: Optional[Dict[str, Any]] = None) -> None:
    """Render a detailed scenario analysis table."""
    
    if not scenario_results:
        return
    
    st.subheader("Detailed Scenario Analysis")
    
    rows = []
    for key, result in scenario_results.items():
        if result is not None:
            if not isinstance(result, dict):
                result = result.to_dict() if hasattr(result, 'to_dict') else {}
            if result:
                rows.append({
                    "Scenario": result.get("scenario_name", key.capitalize()),
                    "EFC/day": result.get("EFC_per_day", ""),
                    "Avg SOC (%)": result.get("Avg_SOC_pct", ""),
                    "Avg DoD (%)": result.get("Avg_DoD_pct", ""),
                    "Mean Temp (°C)": result.get("Mean_Tavg_C", ""),
                    "Max Temp (°C)": result.get("Max_Tmax_C", ""),
                    "Min Temp (°C)": result.get("Min_Tmin_C", ""),
                    "Cycles/day": result.get("Cycles_per_day", ""),
                    "C-rate": result.get("C_rate", ""),
                    "Charge Duration (hr)": result.get("Charge_Duration_hr", ""),
                    "Discharge Duration (hr)": result.get("Discharge_Duration_hr", ""),
                    "EOL (years)": result.get("EOL_year", ""),
                    "EOL Date": result.get("EOL_date", ""),
                    "Final SOH (%)": result.get("Final_SOH_pct", ""),
                    "Calendar Loss (%)": result.get("calendar_loss", ""),
                    "Cycle Loss (%)": result.get("cycle_loss", ""),
                    "Total Loss (%)": result.get("total_loss", ""),
                })
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Scenario Results",
            data=csv,
            file_name="scenario_results.csv",
            mime="text/csv",
        )