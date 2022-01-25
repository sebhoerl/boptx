import numpy as np
import pandas as pd

def calculate_bounds(df, modes, maximum_bin_count, distance = "euclidean_distance", minimum_distance = 0.0, maximum_distance = np.inf):
    bounds = {}

    df = df.copy()
    df = df[df[distance] >= minimum_distance]
    df = df[df[distance] < maximum_distance]

    for mode in modes:
        values = df[distance].values
        cdf = df["weight"].values

        sorter = np.argsort(values)
        values = values[sorter]
        cdf = np.cumsum(cdf[sorter])
        cdf = cdf / cdf[-1]

        probabilities = np.linspace(0.0, 1.0, maximum_bin_count + 1)
        quantiles = np.unique([
            values[np.argmin(cdf <= p)] for p in probabilities
        ])

        bounds[mode] = list(zip(np.arange(len(quantiles)), quantiles[:-1], quantiles[1:]))

    return bounds, distance

def calculate_shares(df_trips, bounds):
    bounds, distance = bounds

    df_shares = []

    if not distance in df_trips:
        raise RuntimeError("Reference has been calculated with %s, but column is not present" % distance)

    # Filter relevant mode
    df_trips = df_trips[df_trips["mode"].isin(list(bounds.keys()))]

    for mode in bounds.keys():
        df_mode = df_trips[df_trips["mode"] == mode]

        for index, lower, upper in bounds[mode]:
            value = df_mode[df_mode["euclidean_distance"].between(lower, upper, inclusive = "left")]["weight"].sum()
            value /= df_trips[df_trips["euclidean_distance"].between(lower, upper, inclusive = "left")]["weight"].sum()

            df_shares.append({
                "mode": mode,
                "bin_index": index,
                "lower_bound": lower,
                "upper_bound": upper,
                "share": value
            })

    return pd.DataFrame.from_records(df_shares)

def calculate_difference(df_reference, df_simulation, bounds):
    df_reference = calculate_shares(df_reference, bounds)

    df_simulation = df_simulation.copy()
    df_simulation["weight"] = 1.0
    df_simulation = calculate_shares(df_simulation, bounds)

    df_difference = pd.merge(
        df_reference.rename(columns = { "share": "reference_share" }),
        df_simulation.rename(columns = { "share": "simulation_share" })[[
            "simulation_share", "mode", "bin_index"
        ]],
        on = ["mode", "bin_index"],
        how = "left"
    ).fillna(0.0)

    df_difference["difference"] = df_difference["simulation_share"] - df_difference["reference_share"]
    return df_difference

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df_reference = pd.read_csv("data/egt_trips.csv", sep = ";")

    bounds = calculate_bounds(
        df = df_reference,
        modes = ["car", "pt", "bike", "walk"],
        maximum_bin_count = 20,
        minimum_distance = 250.0,
        maximum_distance = 40 * 1e3
    )

    df_simulation = pd.read_csv("data/test_output/eqasim_trips.csv", sep = ";")
    df_difference = calculate_difference(df_reference, df_simulation, bounds)

    print(df_difference)

    df_mode = df_difference[df_difference["mode"] == "pt"]
    midpoint = 0.5 * (df_mode["upper_bound"] - df_mode["lower_bound"])

    plt.figure()
    plt.plot(midpoint, df_mode["reference_share"], "--")
    plt.plot(midpoint, df_mode["simulation_share"])
    plt.show()
