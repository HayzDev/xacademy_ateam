import matplotlib.pyplot as plt
from geometry import latlon_to_nm

def plot_map(iceberg_xy, track_end_xy, platforms, ref_lat):
    ix, iy = iceberg_xy
    x2, y2 = track_end_xy

    plt.figure()
    plt.plot([ix, x2], [iy, y2], label="Iceberg Track")

    for name, p in platforms.items():
        px, py = latlon_to_nm(p["lat"], p["lon"], ref_lat)
        plt.scatter(px, py)
        plt.text(px, py, name)

    plt.xlabel("Nautical Miles (E/W)")
    plt.ylabel("Nautical Miles (N/S)")
    plt.axis("equal")
    plt.legend()
    plt.show()