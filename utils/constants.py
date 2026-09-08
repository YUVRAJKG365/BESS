# Physical and model constants for BESS degradation


# Temperature reference points (Celsius)
T_REF = 25.0  # Reference temperature for Arrhenius calculations
T_K_REF = 298.15  # Reference temperature in Kelvin

# Default EOL threshold
EOL_SOH_THRESHOLD = 65.0  # Percentage

# Default calibration bounds
PARAM_BOUNDS = {
    "A_cal": (1e-6, 1e-2),
    "A_cyc": (1e-6, 1e-2),
    "Ea_cal": (1e3, 1e5),  # J/mol
    "Ea_cyc": (1e3, 1e5),  # J/mol
    "alpha": (0.1, 3.0),
    "beta": (0.1, 3.0),
    "k_SOC_high": (0.1, 5.0),
    "k_SOC_reward": (0.1, 3.0),
    "k_SOC_low": (0.1, 5.0),
    "gamma_DoD": (0.1, 5.0),
    "k_C": (0.01, 10.0),
    "k_Tmax": (0.1, 10.0),
    "k_Tmin": (0.1, 10.0),
}

# Stress factor categories
STRESS_CATEGORIES = ["calendar", "cycle", "temperature", "SOC", "DoD", "C-rate", "EFC"]

# Chart colors (professional engineering palette)
CHART_COLORS = {
    "primary": "#2c3e50",
    "secondary": "#34495e", 
    "accent": "#e74c3c",
    "error": "#e74c3c",
    "success": "#27ae60",
    "warning": "#f39c12",
    "info": "#3498db",
    "neutral": "#95a5a6",
}

# Status semantics
STATUS_SEMANTICS = {
    "valid": {"color": "#27ae60", "label": "Valid"},
    "warning": {"color": "#f39c12", "label": "Warning"},
    "error": {"color": "#e74c3c", "label": "Error"},
}