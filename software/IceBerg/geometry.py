import math

def latlon_to_nm(lat, lon, ref_lat):
    x = lon * 60 * math.cos(math.radians(ref_lat))
    y = lat * 60
    return x, y


def iceberg_track(start_x, start_y, heading_deg, length=1000):
    rad = math.radians(heading_deg)
    dx = math.sin(rad)
    dy = math.cos(rad)

    end_x = start_x + dx * length
    end_y = start_y + dy * length
    return end_x, end_y


def point_to_line_distance(px, py, x1, y1, x2, y2):
    num = abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1)
    den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return num / den