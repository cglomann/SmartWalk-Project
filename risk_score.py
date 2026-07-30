def calculate_risk_score(p_cross, p_path_conflict, s_vehicle, c_environment=0.5):
    """
    p_cross: crossing probability (0-1), from LSTM
    p_path_conflict: 1.0 if check_conflict() is True, else 0.0
    s_vehicle: normalized vehicle speed/TTC risk (0-1) -- define the normalization
    c_environment: placeholder until weather and signal data exist, default neutral 0.5
    """
    return(
        0.4 * p_cross + 
        0.3 * p_path_conflict +
        0.2 * s_vehicle +
        0.1 * c_environment
    )

# Tune max_expected_speed once real vehicle speed numbers are seen from video (pixel/sec value)
def normalize_vehicle_risk(vehicle_speed, max_expected_speed=300):
    """Simple normalization of pixel-speed into a 0-1 risk contribution."""
    return min(vehicle_speed / max_expected_speed, 1.0)

