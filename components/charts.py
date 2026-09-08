import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from utils.session_state import SessionStateManager
from utils.constants import CHART_COLORS, EOL_SOH_THRESHOLD


def create_soh_over_time(
    historical_dates: np.ndarray, 
    historical_soh: np.ndarray, 
    forecast_dates: Optional[np.ndarray] = None,
    forecast_soh: Optional[np.ndarray] = None,
    oem_dates: Optional[np.ndarray] = None,
    oem_soh: Optional[np.ndarray] = None,
    title: str = "SOH Over Time",
    show_eol: bool = True,
    eol_threshold: float = EOL_SOH_THRESHOLD,
    uncertainty_bands: Optional[Dict[str, np.ndarray]] = None,
    forecast_dates_arr: Optional[np.ndarray] = None
) -> go.Figure:
    """Create SOH over time chart with optional forecast and OEM baseline."""
    
    fig = go.Figure()
    
    # Historical observed SOH
    if len(historical_dates) > 0 and len(historical_soh) > 0:
        fig.add_trace(go.Scatter(
            x=list(historical_dates),
            y=list(historical_soh),
            mode='lines+markers',
            name='Observed SOH',
            line=dict(color=CHART_COLORS['primary'], width=2),
            marker=dict(size=4, color=CHART_COLORS['primary']),
            hovertemplate='<b>Observed SOH</b><br>Date: %{x|%Y-%m-%d}<br>SOH: %{y:.2f}%<extra></extra>'
        ))
    
    # Forecast SOH
    if forecast_dates is not None and forecast_soh is not None and len(forecast_soh) > 0:
        fig.add_trace(go.Scatter(
            x=list(forecast_dates),
            y=list(forecast_soh),
            mode='lines',
            name='Forecast SOH',
            line=dict(color=CHART_COLORS['accent'], width=3, dash='dash'),
            hovertemplate='<b>Forecast SOH</b><br>Date: %{x|%Y-%m-%d}<br>SOH: %{y:.2f}%<extra></extra>'
        ))
    
    # Uncertainty bands
    if uncertainty_bands and forecast_dates_arr is not None:
        # P10 and P90 for shading
        if 'P10' in uncertainty_bands and 'P90' in uncertainty_bands:
            fig.add_trace(go.Scatter(
                x=list(forecast_dates_arr),
                y=list(uncertainty_bands['P90']),
                mode='lines',
                name='P90 Upper Bound',
                line=dict(width=0),
                showlegend=True,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=list(forecast_dates_arr),
                y=list(uncertainty_bands['P10']),
                fill='tonexty',
                fillcolor='rgba(231, 76, 60, 0.1)',
                line=dict(width=0),
                name='P10-P90 Uncertainty',
                hoverinfo='skip'
            ))
            # P5 and P95
            fig.add_trace(go.Scatter(
                x=list(forecast_dates_arr),
                y=list(uncertainty_bands['P95']),
                mode='lines',
                name='P95 Upper Bound',
                line=dict(width=0),
                showlegend=True,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=list(forecast_dates_arr),
                y=list(uncertainty_bands['P5']),
                fill='tonexty',
                fillcolor='rgba(231, 76, 60, 0.05)',
                line=dict(width=0),
                name='P5-P95 Uncertainty',
                hoverinfo='skip'
            ))
    
    # OEM baseline
    if oem_dates is not None and oem_soh is not None and len(oem_soh) > 0:
        fig.add_trace(go.Scatter(
            x=list(oem_dates),
            y=list(oem_soh),
            mode='lines',
            name='OEM Baseline',
            line=dict(color=CHART_COLORS['info'], width=2, dash='dot'),
            hovertemplate='<b>OEM Baseline</b><br>Date: %{x|%Y-%m-%d}<br>SOH: %{y:.2f}%<extra></extra>'
        ))
    
    # EOL threshold line
    if show_eol and eol_threshold is not None:
        fig.add_hline(
            y=eol_threshold,
            line_dash="dash",
            line_color=CHART_COLORS['warning'],
            annotation_text=f"EOL Threshold: {eol_threshold}%",
            annotation_position="right",
            annotation_font_size=12,
            annotation_font_color=CHART_COLORS['warning']
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#2c3e50')),
        xaxis_title="Date",
        yaxis_title="SOH (%)",
        template="plotly_white",
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor='white',
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#ecf0f1')
    fig.update_yaxes(range=[0, 105])
    
    return fig


def create_scatter_matrix(df: pd.DataFrame, columns: List[str]) -> go.Figure:
    """Create a scatter matrix for exploratory data analysis."""
    if len(columns) < 2:
        return go.Figure()
    
    fig = px.scatter_matrix(
        df,
        dimensions=columns[:6] if len(columns) > 6 else columns,
        title="Feature Scatter Matrix",
        width=800,
        height=800,
        diag_visible=True,
    )
    
    fig.update_layout(
        title=dict(text="Feature Scatter Matrix", font=dict(size=18)),
        template="plotly_white",
    )
    
    return fig


def create_distribution_histogram(
    values: np.ndarray,
    title: str = "Distribution",
    x_title: str = "Value",
    color: str = "#2980b9",
    nbins: int = 30
) -> go.Figure:
    """Create a formatted histogram with mean and median markers."""
    clean = pd.to_numeric(pd.Series(values), errors='coerce').dropna().values
    fig = go.Figure()
    if len(clean) > 0:
        fig.add_trace(go.Histogram(
            x=clean,
            nbinsx=nbins,
            marker_color=color,
            opacity=0.75,
            name="Distribution",
            hovertemplate="<b>Range</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>"
        ))
        mean_val = float(np.mean(clean))
        median_val = float(np.median(clean))
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#e74c3c", annotation_text=f"Mean: {mean_val:.1f}")
        fig.add_vline(x=median_val, line_dash="dot", line_color="#27ae60", annotation_text=f"Median: {median_val:.1f}")
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#2c3e50')),
        xaxis_title=x_title,
        yaxis_title="Count",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False
    )
    return fig


def create_box_plot(df: pd.DataFrame, column: str, 
                     group_col: Optional[str] = None,
                     title: str = "Distribution") -> go.Figure:
    """Create a box plot for a column."""
    if group_col and group_col in df.columns:
        fig = px.box(df, x=group_col, y=column, title=title)
    else:
        fig = px.box(df, y=column, title=title)
    
    fig.update_layout(
        title=dict(text=title, font_size=18, font_weight='bold'),
        template="plotly_white",
        height=400,
    )
    
    return fig


def create_stress_vs_soh(
    df: pd.DataFrame, 
    stress_col: str, 
    soh_col: str = "soh_pct",
    title: str = "Stress vs SOH"
) -> go.Figure:
    """Create scatter plot showing relationship between a stress variable and SOH."""
    fig = go.Figure()
    
    # Add scatter
    valid = df[[stress_col, soh_col]].dropna()
    if len(valid) > 0:
        fig.add_trace(go.Scatter(
            x=valid[stress_col].values,
            y=valid[soh_col].values,
            mode='markers',
            marker=dict(
                size=6,
                color=valid[soh_col].values,
                colorscale='RdYlGn',
                reversescale=True,
                opacity=0.7,
                showscale=True,
                colorbar=dict(title="SOH %")
            ),
            name="Data",
            hovertemplate=f'<b>Stress</b>: %{{x:.2f}}<br><b>SOH</b>: %{{y:.2f}}%<extra></extra>'
        ))
        
        # Add trend line
        if len(valid) > 2:
            z = np.polyfit(valid[stress_col].values, valid[soh_col].values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid[stress_col].min(), valid[stress_col].max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line,
                y=p(x_line),
                mode='lines',
                name=f"Trend (slope: {z[0]:.4f})",
                line=dict(color=CHART_COLORS['accent'], width=2, dash='dash'),
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=stress_col.replace("_", " ").title(),
        yaxis_title="SOH (%)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: List[str]) -> go.Figure:
    """Create a correlation heatmap."""
    if len(columns) < 2:
        return go.Figure()
    
    corr = df[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdYlGn_r',
        zmin=-1,
        zmax=1,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Correlation Matrix", font=dict(size=18)),
        template="plotly_white",
        height=500,
        margin=dict(l=100, r=100, t=60, b=100),
    )
    
    return fig


def create_scenario_comparison_chart(
    scenarios: Dict[str, Any],
    forecast_dates: Optional[np.ndarray] = None
) -> go.Figure:
    """Create SOH trajectory comparison chart for scenarios."""
    fig = go.Figure()
    
    colors = {
        "best": CHART_COLORS['success'],
        "middle": CHART_COLORS['warning'],
        "worst": CHART_COLORS['error'],
    }
    
    order = ["best", "middle", "worst"]
    for name in order:
        if name not in scenarios:
            continue
        data = scenarios[name]
        f_soh = getattr(data, 'forecast_soh', None) if not isinstance(data, dict) else data.get('forecast_soh')
        f_dates = getattr(data, 'forecast_dates', None) if not isinstance(data, dict) else data.get('forecast_dates')
        if f_dates is None:
            f_dates = forecast_dates
            
        if f_soh is not None and f_dates is not None and len(f_soh) > 0:
            fig.add_trace(go.Scatter(
                x=list(f_dates),
                y=list(f_soh),
                mode='lines',
                name=f"{name.capitalize()} Case",
                line=dict(color=colors.get(name.lower(), CHART_COLORS['neutral']), width=3),
                hovertemplate=f'<b>{name.capitalize()} Case</b><br>Date: %{{x|%Y-%m-%d}}<br>SOH: %{{y:.2f}}%<extra></extra>'
            ))
            
    # Add EOL 65% threshold line
    fig.add_hline(
        y=EOL_SOH_THRESHOLD,
        line_dash="dash",
        line_color=CHART_COLORS['error'],
        annotation_text=f"EOL Threshold ({EOL_SOH_THRESHOLD}%)",
        annotation_position="right",
        annotation_font_size=12,
        annotation_font_color=CHART_COLORS['error']
    )
    
    fig.update_layout(
        title=dict(text="Operational Scenario Comparison (Best vs Middle vs Worst)", font=dict(size=18, color='#2c3e50')),
        xaxis_title="Date",
        yaxis_title="SOH (%)",
        template="plotly_white",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#ecf0f1')
    fig.update_yaxes(range=[50, 105])
    
    return fig


def create_eol_bar_chart(scenario_eols: Dict[str, float]) -> go.Figure:
    """Create EOL comparison bar chart."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(scenario_eols.keys()),
            y=list(scenario_eols.values()),
            marker_color=[CHART_COLORS['success'], CHART_COLORS['warning'], CHART_COLORS['error']],
            text=[f"{v:.1f} years" for v in scenario_eols.values()],
            textposition='outside',
            textfont_size=14,
            textfont_color='#2c3e50',
        )
    ])
    
    fig.update_layout(
        title=dict(text="EOL Comparison", font=dict(size=18)),
        xaxis_title="Scenario",
        yaxis_title="EOL (years)",
        template="plotly_white",
        height=400,
        showlegend=False,
        yaxis_range=[0, max(scenario_eols.values()) * 1.2 if scenario_eols else 10],
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#ecf0f1')
    
    return fig


def create_calibration_convergence_chart(history: List[Dict]) -> go.Figure:
    """Create calibration convergence chart."""
    fig = go.Figure()
    
    iterations = list(range(len(history)))
    costs = [h.get('fun', h.get('cost', 0)) for h in history]
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=costs,
        mode='lines',
        name='Objective Function',
        line=dict(color=CHART_COLORS['primary'], width=2),
    ))
    
    fig.update_layout(
        title=dict(text="Calibration Convergence", font=dict(size=18)),
        xaxis_title="Iteration",
        yaxis_title="Objective (RMSE)",
        template="plotly_white",
        height=400,
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#ecf0f1')
    
    return fig


def create_parameter_ensemble_chart(
    eol_percentiles: Dict[str, float]
) -> go.Figure:
    """Create EOL percentile visualization."""
    percentiles = ['P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95']
    values = [eol_percentiles.get(p, 0) for p in percentiles]
    
    fig = go.Figure(data=[
        go.Bar(
            x=percentiles,
            y=values,
            marker_color=[
                CHART_COLORS['error'], CHART_COLORS['error'], 
                CHART_COLORS['warning'], CHART_COLORS['primary'],
                CHART_COLORS['warning'], CHART_COLORS['success'], 
                CHART_COLORS['success']
            ],
            text=[f"{v:.1f}" for v in values],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title=dict(text="EOL Percentile Distribution", font=dict(size=18)),
        xaxis_title="Percentile",
        yaxis_title="EOL (years)",
        template="plotly_white",
        height=400,
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#ecf0f1')
    
    return fig