import os, sys, shutil

# Run script updated from convergence branch
# - data/ile_de_france-1.3.1.jar

# Scenario:
# - data/pc_*

# Reference data updated from previous project:
### - data/daily_flow.csv
# - data/egt_pc_trips.csv
### - data/uber_daily.csv
### - data/uber_zones.gpkg

import logging
logging.basicConfig(level = logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")

# Clean up working directory
if os.path.exists("working_directory"):
    shutil.rmtree("working_directory")
os.mkdir("working_directory")

# Define the objective
from boptx.eqasim.objectives import ModeShareObjective
mode_share_objective = ModeShareObjective("data/egt_pc_trips.csv", dict(
    modes = ["car", "pt", "bike", "walk"],
    maximum_bin_count = 20
))

#from boptx.eqasim.objectives import FlowObjective
#flow_objective = FlowObjective(
#    "data/daily_flow.csv", minimum_count = 10)

#from boptx.eqasim.objectives import TravelTimeObjective
#travel_time_objective = TravelTimeObjective(
#    "data/uber_daily.csv", "data/uber_zones.gpkg", minimum_observations = 10)

from boptx.eqasim.objectives import StuckAgentsObjective, WeightedSumObjective
stuck_objective = StuckAgentsObjective()

from boptx.eqasim.objectives import WeightedSumObjective
sum_objective = WeightedSumObjective()
sum_objective.add("mode_share", 1.0, mode_share_objective, True)
#sum_objective.add("flow", 0.0, flow_objective)
#sum_objective.add("travel_time", 0.0, travel_time_objective)
sum_objective.add("stuck", 1.0, stuck_objective)

# Define the parameters
from boptx.eqasim.problem import ModeParameter, CapacityParameter
parameters = [
    ModeParameter("car.alpha_u", initial_value = 0.0),
    ModeParameter("bike.alpha_u", initial_value = 0.0),
    ModeParameter("walk.alpha_u", initial_value = 0.0),
    ModeParameter("car.betaTravelTime_u_min", (-2.0, 0.0), -0.06, 10.0),
    CapacityParameter()
]

# Define the calibration problem
from boptx.eqasim.problem import CalibrationProblem
problem = CalibrationProblem(sum_objective, parameters)

# Define the evaluator
from boptx.matsim import MATSimEvaluator
evaluator = MATSimEvaluator(
    working_directory = "working_directory", # Working directory, which must exist.
    problem = problem,
    parallelism = 4,
    settings = dict(
        class_path = os.path.realpath("data/ile_de_france-1.3.1.jar"),
        main_class = "org.eqasim.ile_de_france.RunSimulation",
        vm_arguments = ["-Dmatsim.preferLocalDtds=true"],
        memory = "12g",
        config = {
            "global.numberOfThreads": 8,
            "qsim.numberOfThreads": 8,
            "linkStats.writeLinkStatsInterval": 0,
            "controler.writeTripsInterval": 0,
            "controler.lastIteration": 400
        },
        arguments = [
            "--config-path", os.path.realpath("data/pc_config.xml"),
            # "--count-links", os.path.realpath("data/daily_flow.csv"),
        ]
    )
)

# Define the algorithm
from boptx.algorithms import CMAESAlgorithm
algorithm = CMAESAlgorithm(problem)

# Prepare the calibration loop
from boptx.loop import Loop
loop = Loop(
    algorithm = algorithm,
    evaluator = evaluator,
    maximum_evaluations = 10000,
)

# Prepare tracking of the calibration
from boptx.tracker import PickleTracker
loop.advance(callback = PickleTracker("calibration_distance_mode_share.p"))
