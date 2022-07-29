import pickle
import pandas as pd
import geopandas as gpd
import numpy as np

import scipy.stats as ss


import boptx.eqasim.mode_analysis as mode_analysis
import boptx.eqasim.flow_analysis as flow_analysis
import boptx.eqasim.travel_time_analysis as travel_time_analysis

import logging
logger = logging.getLogger(__name__)

class BaseObjective:
    def __init__(self, objective):
        self.objective = objective

    def calculate_objective_(self, values):
        if len(values) == 0.0:
            logger.warn("Calculating objective without any values, falling back to 0.0")
            return 0.0

        if self.objective.upper() == "L1":
            return np.sum(values) / len(values)
        elif self.objective.upper() == "L2":
            return np.sqrt(np.sum(values**2)) / len(values)
        elif self.objective.upper() == "SUP":
            return np.max(values)
        else:
            raise RuntimeError("Unknown objective")

    def get_state_count(self):
        return 0

class ModeShareObjective(BaseObjective):
    def __init__(self, reference_path, bounds_parameters = {}, area = "all", threshold = 0.0, objective = "L1", shares_path = None):
        super().__init__(objective)

        self.area = area
        self.objective = objective
        self.threshold = threshold
        self.shares = None

        self.df_reference = pd.read_csv(reference_path, sep = ";")
        if not "weight" in self.df_reference and "trip_weight" in self.df_reference:
            self.df_reference["weight"] = self.df_reference["trip_weight"]

        self.df_reference = self.filter_area_(self.df_reference)
        self.bounds = mode_analysis.calculate_bounds(self.df_reference, **bounds_parameters)

        if not shares_path is None:
            with open(shares_path, "rb") as f:
                bounds, distance, shares = pickle.load(f)
                self.bounds = (bounds, distance)
                self.shares = shares

    def filter_area_(self, df):
        if self.area == "all":
            return df
        elif self.area == "urban":
            return df[df["urban_origin"] & df["urban_destination"]]
        elif self.area == "non-urban":
            return df[~df["urban_origin"] & ~df["urban_destination"]]
        else:
            raise RuntimeError("Invalid area")

    def calculate(self, simulation_path):
        df_simulation = pd.read_csv("{}/eqasim_trips.csv".format(simulation_path), sep = ";")
        df_urban = pd.read_csv("{}/urban.csv".format(simulation_path), sep = ";")
        df_simulation = pd.merge(df_simulation, df_urban, on = ["person_id", "person_trip_id"])

        df_simulation = self.filter_area_(df_simulation)

        df_difference = mode_analysis.calculate_difference(self.df_reference, df_simulation, self.bounds)

        if not self.shares is None:
            for mode, shares in self.shares.items():
                for k in range(len(shares)):
                    df_difference.loc[
                        (df_difference["mode"] == mode) & (df_difference["bin_index"] == k),
                        "reference_share"] = shares[k]

            df_difference["difference"] = df_difference["simulation_share"] - df_difference["reference_share"]

        objective = np.abs(df_difference["difference"].values)
        objective = np.maximum(0.0, objective - self.threshold)

        objective = self.calculate_objective_(objective)

        states = df_difference.sort_values([
            "mode", "bin_index"
        ])["simulation_share"].values
        assert len(states) == self.get_state_count()

        return {
            "objective": objective,
            "type": "mode_share",
            "configuration": {
                "threshold": self.threshold,
                "data": df_difference,
                "bounds": self.bounds
            },
            "states": states
        }

    def get_state_count(self):
        return sum([
            len(mode)
            for mode in self.bounds[0].values()
        ])


class GlobalModeShareObjective(BaseObjective):
    def __init__(self, reference = {}, threshold = 0.0, objective = "L1"):
        super().__init__(objective)

        self.reference = reference
        self.objective = objective
        self.threshold = threshold

    def calculate(self, simulation_path):
        # Prepare reference shares
        reference_modes = sorted(self.reference.keys())
        reference = np.array([self.reference[mode] for mode in reference_modes])

        # Read simulation shares
        df_simulation = pd.read_csv("{}/eqasim_trips.csv".format(simulation_path), sep = ";")
        simulation_modes = df_simulation["mode"].unique()
        simulation = np.array([
            np.count_nonzero(df_simulation["mode"] == mode) for mode in simulation_modes])
        simulation = simulation / np.sum(reference)

        # Select simulation shares
        simulation = np.array([
            simulation[simulation_modes.index(mode)]
            if simulation_modes.index(mode) >= 0 else 0.0
            for mode in reference_modes
        ])

        states = np.abs(reference - simulation)
        objective = np.maximum(0.0, states - self.threshold)
        objective = self.calculate_objective_(objective)

        return {
            "objective": objective,
            "type": "global_mode_share",
            "configuration": {
                "threshold": self.threshold,
                "reference": self.reference
            },
            "states": states
        }

    def get_state_count(self):
        return len(self.reference)

class FlowObjective(BaseObjective):
    def __init__(self, reference_path, relative_threshold = 0.0, relative = True, objective = "L1", scaling = False, minimum_count = 0, tags = None):
        super().__init__(objective)

        self.objective = objective
        self.relative_threshold = relative_threshold
        self.relative = relative
        self.scaling = scaling
        self.minimum_count = minimum_count
        self.tags = tags

        self.df_reference = pd.read_csv(reference_path, sep = ";")
        self.is_hourly = "hour" in self.df_reference

        if self.objective.upper() == "STD":
            if not self.relative:
                raise RuntimeError("Standard deviation objective only makes sense with relative error")

            if self.scaling:
                raise RuntimeError("Standard deviation objective does not make sense in combination with absolute scaling")

        if self.objective.upper() == "KENDALL":
            if self.relative:
                raise RuntimeError("Kendall tau objective only makes sense with absolute error")

            if self.scaling:
                raise RuntimeError("Standard deviation objective does not make sense in combination with absolute scaling")

        if self.objective.upper() == "R2":
            if self.relative:
                raise RuntimeError("R2 objective only makes sense with absolute error")

    def calculate(self, simulation_path):
        df_simulation = pd.read_csv("{}/eqasim_counts.csv".format(simulation_path), sep = ";")[[
            "link_id", "hour", "count", "osm", "lanes"
        ]]

        if not self.is_hourly:
            df_simulation = pd.merge(
                df_simulation.drop_duplicates("link_id")[["link_id", "osm", "lanes"]],
                df_simulation[["link_id", "count"]].groupby("link_id").sum().reset_index(),
                on = "link_id", how = "left"
            )

            df_simulation["count"] = df_simulation["count"].fillna(0)

        df_difference, scaling_factor = flow_analysis.calculate_difference(self.df_reference, df_simulation, minimum_count = self.minimum_count, tags = self.tags)
        df_valid = df_difference[df_difference["valid"]]

        objective = np.abs(df_valid["scaled_difference" if self.scaling else "difference"].values)

        # Everything within the threshold is counted as zero
        threshold = df_valid["reference_flow"].values * self.relative_threshold
        objective = np.maximum(0.0, objective - threshold)

        if self.relative:
            f = df_valid["reference_flow"] > 0
            objective = objective[f] / df_valid[f]["reference_flow"].values

        if self.objective.upper() == "STD":
            objective = np.std(objective)
        elif self.objective.upper() == "KENDALL":
            objective = (-ss.kendalltau(df_valid["reference_flow"].values, df_valid["simulation_flow"].values).correlation + 1.0) * 0.5
        elif self.objective.upper() == "R2":
            SStot = np.sum((df_valid["reference_flow"].values - df_valid["reference_flow"].mean())**2)
            SSres = np.sum(objective**2)
            objective = 1 - SSres / SStot
            objective = 1 - objective
        else:
            objective = self.calculate_objective_(objective)

        fields = ["link_id"]
        if "hour" in df_difference: fields += ["hour"]
        states = df_difference.sort_values(fields)["simulation_flow"].values
        assert len(states) == self.get_state_count()

        return {
            "objective": objective,
            "type": "flow",
            "configuration": {
                "relative_threshold": self.relative_threshold,
                "relative": self.relative,
                "scaling_factor": scaling_factor,
                "data": df_difference,
                "minimum_count": self.minimum_count,
            },
            "states": states
        }

    def get_state_count(self):
        return len(self.df_reference)

class TravelTimeObjective(BaseObjective):
    def __init__(self, reference_path, zones_path, absolute_threshold = 0.0, relative_threshold = 0.0, relative = True, objective = "L1", minimum_observations = 0):
        super().__init__(objective)

        self.objective = objective
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.relative = relative
        self.minimum_observations = minimum_observations

        self.df_zones = gpd.read_file(zones_path)
        self.df_reference = pd.read_csv(reference_path, sep = ";")
        self.is_hourly = "hour" in self.df_reference

    def calculate(self, simulation_path):
        df_simulation = pd.read_csv("{}/eqasim_trips.csv".format(simulation_path), sep = ";")
        df_simulation = travel_time_analysis.calculate_travel_times(df_simulation, self.df_zones)
        df_difference = travel_time_analysis.calculate_difference(self.df_reference, df_simulation, minimum_observations = self.minimum_observations)
        df_valid = df_difference[df_difference["valid"]]

        objective = np.abs(df_valid["difference"].values)

        # Everything within the threshold is counted as zero
        threshold = df_valid["reference_travel_time"].values * self.relative_threshold + self.absolute_threshold
        objective = np.maximum(0.0, objective - threshold)

        if self.relative:
            f = df_valid["reference_travel_time"] > 0
            objective = objective[f] / df_valid[f]["reference_travel_time"].values

        objective = self.calculate_objective_(objective)

        fields = ["origin_municipality_id", "destination_municipality_id"]
        if "hour" in df_difference: fields += ["hour"]
        states = df_difference.sort_values(fields)["simulation_travel_time"].values
        assert len(states) == self.get_state_count()

        return {
            "objective": objective,
            "type": "travel_time",
            "configuration": {
                "absolute_threshold": self.absolute_threshold,
                "relative_threshold": self.relative_threshold,
                "relative": self.relative,
                "minimum_observations": self.minimum_observations,
                "data": df_difference
            },
            "states": states
        }

    def get_state_count(self):
        return len(self.df_reference)

class StuckAgentsObjective:
    def __init__(self, relative_threshold = 0.0, absolute_threshold = 0.0, relative = True):
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.relative = relative

    def calculate(self, simulation_path):
        df_stuck = pd.read_csv("{}/stuck_analysis.csv".format(simulation_path), sep = ";")
        stuck = df_stuck["count"].values[-1]
        objective = float(stuck)

        df_urban = pd.read_csv("{}/urban.csv".format(simulation_path), sep = ";")
        number_of_agents = len(df_urban["person_id"].unique())

        threshold = self.relative_threshold * number_of_agents + self.absolute_threshold
        objective = np.maximum(0.0, objective - threshold)

        if self.relative:
            objective /= number_of_agents

        return {
            "objective": objective,
            "type": "stuck",
            "configuration": {
                "stuck": stuck,
                "total": number_of_agents,
                "relative_threshold": self.relative_threshold,
                "absolute_threshold": self.absolute_threshold,
                "relative": self.relative
            },
            "states": np.array([stuck])
        }

    def get_state_count(self):
        return 1

class WeightedSumObjective:
    def __init__(self):
        self.objectives = {}
        self.factors = {}
        self.states = []

    def add(self, identifier, factor, objective, use_states = False):
        self.objectives[identifier] = objective
        self.factors[identifier] = factor

        if use_states:
            self.states.append(identifier)

    def calculate(self, simulation_path):
        objective = 0.0
        components = {}

        for identifier in self.objectives.keys():
            result = self.objectives[identifier].calculate(simulation_path)

            objective += self.factors[identifier] * result["objective"]
            components[identifier] = result

        states = np.hstack([
            components[identifier]["states"]
            for identifier in self.states
        ]) if len(self.states) > 0 else None

        if not states is None:
            assert len(states) == self.get_state_count()

        return {
            "objective": objective,
            "type": "weighted_sum",
            "factors": self.factors,
            "components": components,
            "states": states
        }

    def get_state_count(self):
        count = 0

        for identifier in self.states:
            count += self.objectives[identifier].get_state_count()

        return count

if __name__ == "__main__":
    mode_share_objective = ModeShareObjective(
        "data/egt_trips.csv",
        dict(
            modes = ["car", "pt", "bike", "walk"],
            maximum_bin_count = 20
        ),
        objective = "sup"
    )

    flow_objective = FlowObjective(
        "data/daily_flow.csv",
        objective = "l1",
        scaling = True,
        minimum_count = 10
    )

    travel_time_objective = TravelTimeObjective(
        "data/uber_daily.csv",
        "data/uber_zones.gpkg",
        objective = "l1"
    )

    stuck_objective = StuckAgentsObjective()

    sum_objective = WeightedSumObjective()
    sum_objective.add("mode_share", 0.3, mode_share_objective, True)
    sum_objective.add("flow", 0.3, flow_objective, False)
    sum_objective.add("travel_time", 0.3, travel_time_objective, False)
    sum_objective.add("stuck", 0.1, stuck_objective, True)

    print("mode_share", mode_share_objective.calculate("data/test_output")["objective"])
    print("flow", flow_objective.calculate("data/test_output")["objective"])
    print("travel_time", travel_time_objective.calculate("data/test_output")["objective"])
    print("stuck", stuck_objective.calculate("data/test_output")["objective"])
    print("sum", sum_objective.calculate("data/test_output")["objective"])
