import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as geo

def calculate_travel_times(df_trips, df_municipalities, hourly = True):
    df_city = df_municipalities.dissolve().reset_index()[["geometry"]]

    df_trips = df_trips[[
        "travel_time", "departure_time", "mode",
        "origin_x", "origin_y", "destination_x", "destination_y"
    ]].copy()

    df_trips = df_trips[df_trips["mode"] == "car"]

    df_trips["origin_geometry"] = [
        geo.Point(*xy) for xy in zip(df_trips["origin_x"], df_trips["origin_y"])
    ]

    df_trips["destination_geometry"] = [
        geo.Point(*xy) for xy in zip(df_trips["destination_x"], df_trips["destination_y"])
    ]

    df_trips = gpd.GeoDataFrame(df_trips, geometry = "origin_geometry", crs = "epsg:2154")

    df_trips = df_trips.set_geometry("origin_geometry")
    df_trips = gpd.sjoin(df_trips, df_city, predicate = "within")
    df_trips = df_trips.drop(columns = ["index_right"])

    df_trips = df_trips.set_geometry("destination_geometry")
    df_trips = gpd.sjoin(df_trips, df_city, predicate = "within")
    df_trips = df_trips.drop(columns = ["index_right"])

    df_trips = df_trips.set_geometry("origin_geometry")
    df_trips = gpd.sjoin(df_trips, df_municipalities.rename(columns = {
        "municipality_id": "origin_municipality_id"
    }), predicate = "within")
    df_trips = df_trips.drop(columns = ["index_right"])

    df_trips = df_trips.set_geometry("destination_geometry")
    df_trips = gpd.sjoin(df_trips, df_municipalities.rename(columns = {
        "municipality_id": "destination_municipality_id"
    }), predicate = "within")
    df_trips = df_trips.drop(columns = ["index_right"])

    df_trips["hour"] = df_trips["departure_time"] // 3600

    columns = ["origin_municipality_id", "destination_municipality_id"]
    if hourly: columns += ["hour"]

    df_trips = df_trips.groupby(columns)["travel_time"].aggregate(
        ["mean", "size"]).reset_index().rename(columns = {
        "mean": "travel_time", "size": "observations"
    })

    return df_trips

def calculate_difference(df_reference, df_simulation, minimum_observations = 0):
    columns = ["origin_municipality_id", "destination_municipality_id"]

    df_simulation = df_simulation[df_simulation["observations"] >= minimum_observations]

    if not "hour" in df_reference:
        df_simulation = df_simulation.groupby([
            "origin_municipality_id", "destination_municipality_id"
        ])["travel_time"].mean().reset_index()
    else:
        columns += ["hour"]

    df_simulation = df_simulation.rename(columns = {
        "travel_time": "simulation_travel_time"
    })

    df_reference = df_reference.rename(columns = {
        "travel_time": "reference_travel_time"
    })

    df = pd.merge(df_reference, df_simulation, on = columns, how = "left")

    df["valid"] = ~df["simulation_travel_time"].isna()
    df["simulation_travel_time"] = df["simulation_travel_time"].fillna(0.0)

    df["difference"] = df["simulation_travel_time"] - df["reference_travel_time"]

    return df

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df_simulation = pd.read_csv("data/test_output/trips.csv", sep = ";")
    df_zones = gpd.read_file("data/uber_zones.gpkg")

    df_simulation = calculate_travel_times(df_simulation, df_zones)
    print(df_simulation["observations"].value_counts())

    df_reference = pd.read_csv("data/uber_hourly.csv", sep = ";")
    df_difference = calculate_difference(df_reference, df_simulation, minimum_observations = 10)
    print(df_difference)

    df_difference = df_difference[df_difference["valid"]]
    print(df_difference)
    
    plt.plot(df_difference["reference_travel_time"], df_difference["simulation_travel_time"], 'x')
    plt.plot([0, 7200], [0, 7200], "k")
    plt.show()
