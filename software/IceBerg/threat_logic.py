def surface_threat(distance_nm, keel, water_depth):
    if keel >= 1.10 * water_depth:
        return "GREEN"
    if distance_nm < 5:
        return "RED"
    elif distance_nm <= 10:
        return "YELLOW"
    return "GREEN"


def subsea_threat(keel, water_depth):
    ratio = keel / water_depth

    if ratio >= 1.10:
        return "GREEN"
    elif ratio >= 0.90:
        return "RED"
    elif ratio >= 0.70:
        return "YELLOW"
    return "GREEN"