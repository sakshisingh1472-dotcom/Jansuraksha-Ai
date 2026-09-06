def calculate_risk(density: float) -> str:
    if density >= 85: return "CRITICAL"
    if density >= 70: return "HIGH"
    if density >= 30: return "NORMAL"
    return "LOW"
