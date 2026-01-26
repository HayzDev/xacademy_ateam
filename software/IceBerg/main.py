from input_data import get_platforms, get_iceberg
from geometry import latlon_to_nm, iceberg_track, point_to_line_distance
from threat_logic import surface_threat, subsea_threat
from plotting import plot_map

# INPUT DATA
platforms = get_platforms()
iceberg_lat, iceberg_lon, heading, keel = get_iceberg()

# CONVERT ICEBERG POSITION
ref_lat = iceberg_lat
ix, iy = latlon_to_nm(iceberg_lat, iceberg_lon, ref_lat)
x2, y2 = iceberg_track(ix, iy, heading)

print("\n--- THREAT ANALYSIS ---")

# ANALYZE EACH PLATFORM
for name, p in platforms.items():
    px, py = latlon_to_nm(p["lat"], p["lon"], ref_lat)
    dist = point_to_line_distance(px, py, ix, iy, x2, y2)

    surface = surface_threat(dist, keel, p["depth"])
    subsea = subsea_threat(keel, p["depth"])

    print(f"\n{name}")
    print(f"  Closest approach: {dist:.2f} nm")
    print(f"  Surface threat: {surface}")
    print(f"  Subsea threat: {subsea}")

# PLOT FOR JUDGES
plot_map((ix, iy), (x2, y2), platforms, ref_lat)