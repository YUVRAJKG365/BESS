import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import io
import json

from utils.constants import EOL_SOH_THRESHOLD


@dataclass
class ReportData:
    """Container for all data needed to generate report."""
    project_info: Dict[str, Any] = field(default_factory=dict)
    uploaded_datasets: List[Dict[str, Any]] = field(default_factory=list)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    dataset_statistics: Dict[str, Any] = field(default_factory=dict)
    oem_baseline: Dict[str, Any] = field(default_factory=dict)
    selected_model: str = "Physics-Based"
    calibration_parameters: Dict[str, float] = field(default_factory=dict)
    calibration_warnings: List[str] = field(default_factory=list)
    historical_fit: Dict[str, Any] = field(default_factory=dict)
    forecast: Dict[str, Any] = field(default_factory=dict)
    best_case: Dict[str, Any] = field(default_factory=dict)
    middle_case: Dict[str, Any] = field(default_factory=dict)
    worst_case: Dict[str, Any] = field(default_factory=dict)
    operating_condition_analysis: Dict[str, Any] = field(default_factory=dict)
    model_performance: Dict[str, Any] = field(default_factory=dict)
    model_comparison: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    eol_prediction: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    technical_notes: List[str] = field(default_factory=list)


class ReportGenerator:
    """Generates professional Excel and CSV reports."""
    
    def __init__(self, report_data: ReportData):
        self.data = report_data
    
    def generate_excel_report(self) -> bytes:
        """Generate comprehensive Excel report."""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            self._write_summary_sheet(writer)
            self._write_datasets_sheet(writer)
            self._write_data_quality_sheet(writer)
            self._write_oem_baseline_sheet(writer)
            self._write_calibration_sheet(writer)
            self._write_historical_fit_sheet(writer)
            self._write_forecast_sheet(writer)
            self._write_scenarios_sheet(writer)
            self._write_operating_conditions_sheet(writer)
            self._write_model_performance_sheet(writer)
            self._write_model_comparison_sheet(writer)
            self._write_uncertainty_sheet(writer)
            self._write_assumptions_warnings_sheet(writer)
        
        return output.getvalue()
    
    def _write_summary_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write executive summary sheet."""
        fc = self.data.forecast
        final_soh_val = fc.get('final_soh')
        if final_soh_val is None and 'forecast_soh' in fc and len(fc['forecast_soh']) > 0:
            final_soh_val = float(fc['forecast_soh'][-1])
            
        summary_data = [
            ["BESS SOH Forecasting & Degradation Analytics Report", ""],
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["", ""],
            ["Project Information", ""],
            ["Project Name", self.data.project_info.get("name", "BESS Analysis")],
            ["Location", self.data.project_info.get("location", "N/A")],
            ["System Capacity", self.data.project_info.get("capacity", "N/A")],
            ["Chemistry", self.data.project_info.get("chemistry", "N/A")],
            ["", ""],
            ["Key Results", ""],
            ["Selected Model", self.data.selected_model],
            ["Current SOH (%)", str(self.data.historical_fit.get('current_soh', 'N/A'))],
            ["Predicted EOL (years)", str(self.data.eol_prediction.get('years_to_eol', 'N/A'))],
            ["EOL Date", str(self.data.eol_prediction.get('eol_date', 'N/A'))],
            ["Final Forecast SOH (%)", f"{final_soh_val:.2f}%" if isinstance(final_soh_val, (int, float)) else str(final_soh_val)],
            ["OEM Baseline EOL (years)", str(self.data.oem_baseline.get('eol_years', 'N/A'))],
            ["", ""],
            ["Scenario Comparison", ""],
            ["Best Case EOL (years)", str(self.data.best_case.get('EOL_year', self.data.best_case.get('eol_years', 'N/A')))],
            ["Middle Case EOL (years)", str(self.data.middle_case.get('EOL_year', self.data.middle_case.get('eol_years', 'N/A')))],
            ["Worst Case EOL (years)", str(self.data.worst_case.get('EOL_year', self.data.worst_case.get('eol_years', 'N/A')))],
        ]
        
        df = pd.DataFrame(summary_data, columns=["Parameter", "Value"])
        df.to_excel(writer, sheet_name="Summary", index=False)
    
    def _write_datasets_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write uploaded datasets information."""
        rows = []
        for ds in self.data.uploaded_datasets:
            rows.append({
                "File Name": ds.get("file_name", ""),
                "File Type": ds.get("file_type", ""),
                "Rows": ds.get("rows", 0),
                "Columns": ds.get("columns", 0),
                "Date Range Start": ds.get("date_range_start", ""),
                "Date Range End": ds.get("date_range_end", ""),
                "Detected SOH Column": ds.get("detected_soh", ""),
                "Detected Temperature Columns": ", ".join(ds.get("detected_temp", [])),
                "Detected SOC Column": ds.get("detected_soc", ""),
                "Detected DoD Column": ds.get("detected_dod", ""),
                "Detected EFC Column": ds.get("detected_efc", ""),
            })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name="Uploaded Datasets", index=False)
    
    def _write_data_quality_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write data quality validation results."""
        dq = self.data.data_quality
        rows = [
            ["Validation Category", "Status", "Details"],
            ["Schema Validation", "Passed" if dq.get("schema_ok") else "Failed", "All required physics columns identified"],
            ["Data Consistency", "Passed" if not dq.get("warnings") else "Warnings Present", f"{len(dq.get('warnings', []))} data quality notes"],
        ]
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df.to_excel(writer, sheet_name="Data Quality", index=False)
        
    def _write_oem_baseline_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write OEM baseline information."""
        oem = self.data.oem_baseline
        rows = [["Parameter", "Value"]]
        for k, v in oem.items():
            if not isinstance(v, (list, np.ndarray, pd.Series)):
                rows.append([k, v])
        if len(rows) > 1:
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df.to_excel(writer, sheet_name="OEM Baseline", index=False)
            
    def _write_calibration_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write calibration parameters."""
        params = self.data.calibration_parameters
        rows = [["Parameter", "Value"]]
        for k, v in params.items():
            rows.append([k, v])
        if len(rows) > 1:
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df.to_excel(writer, sheet_name="Calibration", index=False)
    
    def _write_historical_fit_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write historical model fit data."""
        hf = self.data.historical_fit
        fc = self.data.forecast
        
        dates = hf.get("dates", fc.get("historical_dates", []))
        observed = hf.get("observed", fc.get("historical_soh", []))
        predicted = hf.get("predicted", fc.get("historical_fit", []))
        
        if len(dates) > 0 and len(observed) > 0 and len(predicted) > 0:
            min_len = min(len(dates), len(observed), len(predicted))
            obs_arr = np.array(observed[:min_len], dtype=float)
            pred_arr = np.array(predicted[:min_len], dtype=float)
            df = pd.DataFrame({
                "Date": dates[:min_len],
                "Observed SOH (%)": obs_arr,
                "Predicted SOH (%)": pred_arr,
                "Residual (%)": obs_arr - pred_arr,
            })
            df.to_excel(writer, sheet_name="Historical Fit", index=False)
        
        # Metrics
        metrics = hf.get("metrics", fc.get("metrics", {}))
        if metrics:
            rows = [["Metric", "Value"]]
            for k, v in metrics.items():
                rows.append([k, v])
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df.to_excel(writer, sheet_name="Fit Metrics", index=False)
    
    def _write_forecast_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write forecast data."""
        fc = self.data.forecast
        dates = fc.get("forecast_dates", fc.get("dates", []))
        soh = fc.get("forecast_soh", fc.get("soh", []))
        cal_loss = fc.get("calendar_loss_forecast", fc.get("calendar_loss", []))
        cyc_loss = fc.get("cycle_loss_forecast", fc.get("cycle_loss", []))
        
        if len(dates) > 0 and len(soh) > 0:
            min_len = min(len(dates), len(soh))
            data_dict = {
                "Date": dates[:min_len],
                "Forecast SOH (%)": soh[:min_len],
            }
            if len(cal_loss) >= min_len:
                data_dict["Calendar Loss (%)"] = cal_loss[:min_len]
            if len(cyc_loss) >= min_len:
                data_dict["Cycle Loss (%)"] = cyc_loss[:min_len]
                if "Calendar Loss (%)" in data_dict:
                    data_dict["Total Loss (%)"] = np.array(cal_loss[:min_len]) + np.array(cyc_loss[:min_len])
            df = pd.DataFrame(data_dict)
            df.to_excel(writer, sheet_name="Forecast", index=False)
        
        # EOL info
        eol = self.data.eol_prediction
        if eol:
            rows = [["Parameter", "Value"]]
            for k, v in eol.items():
                rows.append([k, v])
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df.to_excel(writer, sheet_name="EOL Prediction", index=False)
    
    def _write_scenarios_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write scenario comparison."""
        scenarios = [
            ("Best Case", self.data.best_case),
            ("Middle Case", self.data.middle_case),
            ("Worst Case", self.data.worst_case),
        ]
        
        rows = [["Scenario", "EFC/day", "Avg SOC (%)", "Avg DoD (%)", "Mean Temp (°C)", 
                 "Max Temp (°C)", "Cycles/day", "C-rate", "EOL (years)", "Final SOH (%)",
                 "Cal. Loss", "Cyc. Loss", "Total Loss"]]
        
        for name, sc in scenarios:
            if hasattr(sc, 'to_dict'):
                sc = sc.to_dict()
            elif not isinstance(sc, dict):
                sc = {}
            rows.append([
                name,
                sc.get("EFC_per_day", ""),
                sc.get("Avg_SOC_pct", ""),
                sc.get("Avg_DoD_pct", ""),
                sc.get("Mean_Tavg_C", ""),
                sc.get("Max_Tmax_C", ""),
                sc.get("Cycles_per_day", ""),
                sc.get("C_rate", ""),
                sc.get("EOL_year", ""),
                sc.get("Final_SOH_pct", ""),
                sc.get("calendar_loss", ""),
                sc.get("cycle_loss", ""),
                sc.get("total_loss", ""),
            ])
        
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df.to_excel(writer, sheet_name="Scenarios", index=False)
    
    def _write_operating_conditions_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write operating condition impact analysis."""
        oca = self.data.operating_condition_analysis
        
        if "correlations" in oca:
            rows = [["Variable", "Correlation with SOH", "P-value", "Interpretation"]]
            for var, stats in oca["correlations"].items():
                rows.append([var, stats.get("correlation", ""), stats.get("p_value", ""), 
                           stats.get("interpretation", "")])
            
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df.to_excel(writer, sheet_name="Operating Conditions", index=False)
    
    def _write_model_performance_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write model performance metrics."""
        mp = self.data.model_performance
        
        rows = [["Metric", "Value"]]
        for k, v in mp.items():
            rows.append([k, v])
        
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df.to_excel(writer, sheet_name="Model Performance", index=False)
    
    def _write_model_comparison_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write model comparison (Physics vs PINN)."""
        mc = self.data.model_comparison
        
        if "comparison_table" in mc:
            df = pd.DataFrame(mc["comparison_table"])
            df.to_excel(writer, sheet_name="Model Comparison", index=False)
    
    def _write_uncertainty_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write uncertainty analysis results."""
        unc = self.data.uncertainty
        
        if "eol_percentiles" in unc:
            rows = [["Percentile", "EOL (years)"]]
            for p, v in unc["eol_percentiles"].items():
                rows.append([p, v])
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df.to_excel(writer, sheet_name="EOL Uncertainty", index=False)
        
        if "soh_bands" in unc and "forecast_dates" in unc:
            df = pd.DataFrame({"Date": unc["forecast_dates"]})
            for p, band in unc["soh_bands"].items():
                df[f"SOH {p} (%)"] = band
            df.to_excel(writer, sheet_name="SOH Uncertainty Bands", index=False)
    
    def _write_assumptions_warnings_sheet(self, writer: pd.ExcelWriter) -> None:
        """Write assumptions, warnings, and technical notes."""
        rows = [["Category", "Description"]]
        
        for a in self.data.assumptions:
            rows.append(["Assumption", a])
        for w in self.data.warnings:
            rows.append(["Warning", w])
        for n in self.data.technical_notes:
            rows.append(["Technical Note", n])
        
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df.to_excel(writer, sheet_name="Assumptions & Notes", index=False)


def generate_report(report_data: ReportData) -> bytes:
    """Convenience function to generate Excel report."""
    generator = ReportGenerator(report_data)
    return generator.generate_excel_report()


def export_forecast_csv(report_data: ReportData) -> str:
    """Export forecast as CSV string."""
    fc = report_data.forecast
    if "dates" in fc and "soh" in fc:
        df = pd.DataFrame({
            "Date": fc["dates"],
            "SOH (%)": fc["soh"],
        })
        return df.to_csv(index=False)
    return ""


def export_scenarios_csv(report_data: ReportData) -> str:
    """Export scenarios as CSV string."""
    scenarios = [
        ("Best Case", report_data.best_case),
        ("Middle Case", report_data.middle_case),
        ("Worst Case", report_data.worst_case),
    ]
    
    rows = []
    for name, sc in scenarios:
        rows.append({
            "Scenario": name,
            "EFC_per_day": sc.get("EFC_per_day", ""),
            "Avg_SOC_pct": sc.get("Avg_SOC_pct", ""),
            "Avg_DoD_pct": sc.get("Avg_DoD_pct", ""),
            "Mean_Tavg_C": sc.get("Mean_Tavg_C", ""),
            "Max_Tmax_C": sc.get("Max_Tmax_C", ""),
            "Cycles_per_day": sc.get("Cycles_per_day", ""),
            "C_rate": sc.get("C_rate", ""),
            "EOL_years": sc.get("EOL_year", ""),
            "Final_SOH_pct": sc.get("Final_SOH_pct", ""),
            "Calendar_Loss": sc.get("calendar_loss", ""),
            "Cycle_Loss": sc.get("cycle_loss", ""),
            "Total_Loss": sc.get("total_loss", ""),
        })
    
    if rows:
        return pd.DataFrame(rows).to_csv(index=False)
    return ""


def export_calibration_csv(report_data: ReportData) -> str:
    """Export calibration parameters as CSV string."""
    params = report_data.calibration_parameters
    if params:
        df = pd.DataFrame([params])
        return df.to_csv(index=False)
    return ""