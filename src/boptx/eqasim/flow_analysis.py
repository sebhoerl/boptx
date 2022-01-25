import numpy as np
import pandas as pd
import scipy.optimize as opt

def calculate_difference(df_reference, df_simulation, factor = 1.0, minimum_count = 0.0):
    columns = ["link_id"]

    if "hour" in df_reference:
        columns.append("hour")

    if "hour" in df_reference and "hour" in df_simulation:
        pass # Hourly
    elif not "hour" in df_reference and not "hour" in df_simulation:
        pass # Daily
    else:
        raise RuntimeError("Either hourly or daily data must be given for both reference and simulation")

    df_simulation = df_simulation.rename(columns = {
        "count": "simulation_flow"
    })

    df_reference = df_reference.rename(columns = {
        "flow": "reference_flow"
    })

    df = pd.merge(
        df_reference[columns + ["reference_flow"]],
        df_simulation[columns + ["simulation_flow"]],
        on = columns, how = "left"
    )

    df["simulation_flow"] = df["simulation_flow"].fillna(0.0)

    df["valid"] = df["simulation_flow"] >= minimum_count
    df["simulation_flow"] *= factor

    df["difference"] = df["simulation_flow"] - df["reference_flow"]

    # Scaling
    problem = ScalingProblem(
        df[df["valid"]]["reference_flow"].values,
        df[df["valid"]]["simulation_flow"].values
    )

    s = problem.solve()
    df["scaled_difference"] = df["simulation_flow"] * s - df["reference_flow"]
    df["scaled_flow"] = df["simulation_flow"] * s

    return df, s

class ScalingProblem:
    def __init__(self, reference_values, simulation_values):
        self.reference_values = reference_values
        self.simulation_values = simulation_values

    def calculate_objective(self, s):
        return np.sum((self.simulation_values * s[0] - self.reference_values)**2) / len(self.simulation_values)

    def calculate_derivative(self, s):
        return np.sum(2 * (self.simulation_values * s[0] - self.reference_values) * self.simulation_values) / len(self.simulation_values)

    def solve(self):
        result = opt.minimize(
            fun = self.calculate_objective,
            jac = self.calculate_derivative,
            x0 = [1.0]
        )

        if not result.success:
            print("Optimization failed, this should not happen!")
            return 1.0

        return result.x[0]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df_reference = pd.read_csv("data/daily_flow.csv", sep = ";")
    df_simulation = pd.read_csv("data/test_output/eqasim_counts.csv", sep = ";")
    df_simulation = df_simulation[[
        "link_id", "count"
    ]].groupby(["link_id"]).sum().reset_index()
    df_difference, s = calculate_difference(df_reference, df_simulation, 1e3, minimum_count = 10)

    print(df_difference)
    print(s)
    #df_selection = df_difference[df_difference["link_id"] == 78096]
    #plt.plot(df_selection["hour"], df_selection["reference_flow"])
    #plt.plot(df_selection["hour"], df_selection["simulation_flow"])
    #plt.plot(df_selection["hour"], df_selection["scaled_flow"])
    #plt.show()

    df_difference = df_difference[df_difference["valid"]]
    print(df_difference)

    plt.plot(df_difference["reference_flow"], df_difference["simulation_flow"], 'x')
    plt.plot(df_difference["reference_flow"], df_difference["scaled_flow"], 'x')
    plt.plot([0, 140000], [0, 140000], "k")
    plt.show()
