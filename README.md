# BESS SOH Forecasting & Degradation Analytics

A production-grade Streamlit web application for Battery Energy Storage System (BESS) State-of-Health (SOH) degradation analysis and forecasting.

## 🎯 Overview

This application provides a professional analytics platform for BESS degradation modeling, combining:

- **Physics-Based Degradation Model** - Semi-empirical formulation with Arrhenius temperature dependence, SOC/DoD stress factors, and two-stage calibration
- **PINN-Based Model** - Physics-Informed Neural Network approach (interface ready)
- **Multi-File Upload** - Upload multiple field data datasets
- **Scenario Analysis** - Best, middle, and worst case forecasts
- **Uncertainty Analysis** - Parameter ensemble analysis
- **Model Comparison** - Physics vs PINN comparison

## 📁 Project Structure

```
BESS/
├── app.py                    # Main Streamlit application
├── pages/
│   ├── 01_data_upload.py     # Data upload page
│   ├── 02_data_validation.py # Data validation page
│   ├── 03_data_exploration.py# Data exploration page
│   ├── 04_oem_baseline.py    # OEM baseline page
│   ├── 05_model_selection.py # Model selection page
│   ├── 06_calibration.py     # Calibration page
│   ├── 07_forecasting.py     # Forecasting page
│   ├── 08_scenario_analysis.py # Scenario analysis page
│   └── 09_final_report.py    # Final report page
├── components/               # Streamlit UI components
│   ├── sidebar.py            # Sidebar navigation
│   ├── cards.py              # KPI and status cards
│   ├── charts.py             # Plotly chart functions
│   ├── tables.py             # Data table renderers
│   ├── upload.py             # File upload components
│   ├── progress.py           # Progress indicators
│   └── scenario_cards.py     # Scenario card components
├── backend/                  # Backend model modules
│   ├── data_loader.py        # Data loading and metadata extraction
│   ├── data_validator.py     # Data validation logic
│   ├── data_mapper.py        # Column mapping
│   ├── oem_processor.py      # OEM baseline processing
│   ├── physics_model.py      # Physics-based degradation model
│   ├── pinn_model.py         # PINN model interface
│   ├── calibration.py        # Two-stage calibration
│   ├── forecasting.py        # Forecasting engine
│   ├── scenarios.py          # Scenario analysis
│   ├── uncertainty.py        # Uncertainty analysis
│   ├── diagnostics.py        # Diagnostic tools
│   └── report_generator.py   # Excel/CSV report generation
├── utils/                    # Utility modules
│   ├── session_state.py      # Streamlit session state management
│   ├── formatting.py         # HTML formatting utilities
│   ├── constants.py          # Physical and model constants
│   └── helpers.py            # Helper functions
├── outputs/                  # Output directory
├── data/                     # Data directory
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BESS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📋 Workflow

1. **Data Upload** - Upload multiple BESS field data files (Excel/CSV) and an OEM baseline file
2. **Data Validation** - Validate uploaded datasets against model requirements
3. **Data Exploration** - Visualize data spread, distributions, and relationships
4. **OEM Baseline** - Upload/select OEM baseline and view comparison
5. **Model Selection** - Choose Physics-Based or PINN-Based model
6. **Calibration** - Run two-stage calibration (global + local optimization)
7. **Forecast** - Generate SOH forecasts with historical fit
8. **Scenario Analysis** - Generate best/middle/worst case scenarios
9. **Model Comparison** - Compare Physics-Based vs PINN-Based models
10. **Final Report** - Generate comprehensive Excel report

## 📊 Input File Format

### Field Data Files
Upload Excel (.xlsx, .xls) or CSV files containing:

- **SOH (%)** - Battery State of Health percentage
- **Temperature (°C)** - Cell or ambient temperature
- **SOC (%)** - State of Charge
- **DoD (%)** - Depth of Discharge
- **EFC/day** - Equivalent Full Cycles per day
- **Cycles/day** - Number of cycles per day
- **Charge Duration (hr)** - Charge duration in hours
- **Discharge Duration (hr)** - Discharge duration in hours
- **C-rate** - Charge/discharge rate
- **Time/Date** - Timestamp column

### OEM Baseline Files
Upload a file containing historical SOH data from the manufacturer or reference system.

## 🔬 Physics-Based Model

The physics-based degradation model implements:

- **Calendar Aging**: `ΔSOH_cal = A_cal * t^α * exp(-Ea/R*(1/T-1/Tref)) * f_SOC(SOC)`
- **Cycle Aging**: `ΔSOH_cyc = A_cyc * (EFC)^β * f_DoD(DoD) * f_T(T) * f_SOC(SOC) * f_C(C)`
- **Arrhenius Temperature Dependence**: Temperature acceleration factor
- **SOC Stress**: High SOC (>80%) and low SOC (<20%) acceleration
- **DoD Stress**: Power-law dependence on depth of discharge
- **C-Rate Stress**: Exponential dependence on C-rate
- **Tmax/Tmin Stress**: Peaks-over-threshold extreme temperature handling
- **Power-Law Time/Throughput**: Time and energy throughput dependence

## 🧠 PINN Model

The PINN model interface provides:

- **Data Loss**: Match observed SOH values
- **Physics Loss**: Satisfy degradation ODE constraints
- **Boundary Loss**: SOH(0) = 100%, SOH(EOL) = 65%
- **Total Loss**: `L_total = L_data + L_physics + L_boundary`

*Note: The PINN implementation is currently an interface that delegates to the physics model. Full PyTorch/TensorFlow implementation requires additional dependencies.*

## 📈 Features

- **Multi-File Upload**: Upload and manage multiple field data datasets
- **Intelligent Column Mapping**: Auto-detect and manually adjust column mappings
- **Data Validation**: File, schema, date, numeric, missing-value, duplicate, physical plausibility checks
- **Professional Charts**: Interactive Plotly visualizations with zoom, hover, and download
- **Scenario Analysis**: Best/middle/worst case derived from observed distributions
- **Uncertainty Analysis**: Parameter ensemble with P5-P95 percentiles
- **Downloadable Reports**: Excel reports, CSV exports, and plot downloads
- **Error Handling**: Graceful handling of malformed files, missing data, and model failures

## 🧪 Testing

Tests are available for:
- File loading and parsing
- Column mapping
- Data validation
- Scenario generation
- Physics model wrapper
- PINN model wrapper
- Result schema validation
- EOL calculation
- Report generation

## 📝 Assumptions

- SOH degrades monotonically over time (with step events)
- EOL threshold is 65% SOH (configurable)
- Reference temperature is 25°C (298.15K)
- Calendar aging follows power-law time dependence
- Cycle aging follows power-law throughput dependence
- Parameter ensemble is a local sensitivity analysis, not a formal confidence interval

## ⚠️ Important Notes

- All numbers displayed are dynamically calculated from uploaded data
- No hardcoded results - everything comes from actual model calculations
- Do not claim scientific validation that has not been demonstrated
- Uncertainty analysis is labeled as local sensitivity, not statistical confidence

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License

## 📞 Support

For questions or issues, please open an issue on the repository or contact the development team.