def get_platforms():
    platforms = {}
    n = int(input("Enter number of platforms: "))

    for i in range(n):
        print(f"\nPlatform {i+1}")
        name = input("Name: ")
        lat = float(input("Latitude: "))
        lon = float(input("Longitude: "))
        depth = float(input("Water depth (m): "))

        # Convert negative depths to positive automatically
        depth = abs(depth)

        platforms[name] = {
            "lat": lat,
            "lon": lon,
            "depth": depth
        }

    # Review and edit loop
    while True:
        print("\n--- Review Platforms ---")
        for idx, (name, p) in enumerate(platforms.items(), 1):
            print(f"{idx}. {name}: lat={p['lat']}, lon={p['lon']}, depth={p['depth']}")

        edit = input("\nDo you want to edit any platform? (y/n): ").lower()
        if edit != 'y':
            break

        idx_to_edit = int(input("Enter the number of the platform to edit: "))
        key_list = list(platforms.keys())
        edit_name = key_list[idx_to_edit - 1]
        print(f"Editing {edit_name}")
        platforms[edit_name]['lat'] = float(input("New latitude: "))
        platforms[edit_name]['lon'] = float(input("New longitude: "))
        platforms[edit_name]['depth'] = abs(float(input("New depth (m): ")))

    return platforms


def get_iceberg():
    print("\nIceberg Data")
    lat = float(input("Iceberg latitude: "))
    lon = float(input("Iceberg longitude: "))
    heading = float(input("Heading (degrees from north): "))
    keel = float(input("Keel depth (m): "))

    # Convert negative keel depth to positive
    keel = abs(keel)

    # Review and edit iceberg
    while True:
        print(f"\n--- Review Iceberg ---\nLatitude: {lat}, Longitude: {lon}, Heading: {heading}, Keel: {keel}")
        edit = input("Do you want to edit the iceberg data? (y/n): ").lower()
        if edit != 'y':
            break
        lat = float(input("New latitude: "))
        lon = float(input("New longitude: "))
        heading = float(input("New heading (degrees): "))
        keel = abs(float(input("New keel depth (m): ")))

    return lat, lon, heading, keel
