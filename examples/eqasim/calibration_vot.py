import os, sys

import logging
logging.basicConfig(level = logging.INFO)

from objectives import ModeShareObjective, FlowObjective, TravelTimeObjective, StuckAgentsObjective, WeightedSumObjective

selected_objective = None
parallelism = 1
threads = None

if len(sys.argv) > 1:
    selected_objective = sys.argv[1]

if len(sys.argv) > 2:
    parallelism = int(sys.argv[2])

    import multiprocessing
    threads = max(4, multiprocessing.cpu_count() // parallelism)

if len(sys.argv) > 3:
    threads = int(sys.argv[3])

if not selected_objective in ("mode_share", "travel_time", "flow"):
    raise RuntimeError("Unknown objective: {}".format(selected_objective))

mode_share_objective = ModeShareObjective("data/egt_trips.csv", dict(
    modes = ["car", "pt", "bike", "walk"],
    maximum_bin_count = 20
))

flow_objective = FlowObjective("data/daily_flow.csv", minimum_count = 10)
travel_time_objective = TravelTimeObjective("data/uber_daily.csv", "data/uber_zones.gpkg", minimum_observations = 10)
stuck_objective = StuckAgentsObjective()

sum_objective = WeightedSumObjective()
sum_objective.add("mode_share", 1.0 if selected_objective == "mode_share" else 0.0, mode_share_objective, True)
sum_objective.add("flow", 1.0 if selected_objective == "flow" else 0.0, flow_objective)
sum_objective.add("travel_time", 1.0 if selected_objective == "travel_time" else 0.0, travel_time_objective)

stuck_factor = 1.0
if selected_objective == "travel_time": stuck_factor = 10.0

sum_objective.add("stuck", stuck_factor, stuck_objective)

from problem import ModeParameter, CapacityParameter
parameters = [
    ModeParameter("car.alpha_u", (-3.0, 3.0), 0.0),
    ModeParameter("car.betaTravelTime_u_min", (-0.5, 0.0), -0.06, scaling = 10.0),
    ModeParameter("bike.alpha_u", (-3.0, 3.0), 0.0),
    ModeParameter("walk.alpha_u", (-3.0, 3.0), 0.0),
    CapacityParameter()
]

from problem import CalibrationProblem
problem = CalibrationProblem(sum_objective, parameters)

from boptx.algorithms import CMAESAlgorithm
algorithm = CMAESAlgorithm(problem)

from matsim import MATSimEvaluator
evaluator = MATSimEvaluator(
    working_directory = "work", # Working directory, which must exist.
    parameters = parameters,
    objective = sum_objective,
    parallelism = parallelism,
    settings = dict(
        class_path = os.path.realpath("data/ile_de_france.jar"),
        main_class = "org.eqasim.ile_de_france.RunSimulation",
        iterations = 400,
        memory = "24g",
        threads = threads,
        arguments = [
            # "--config-path", os.path.realpath("data/sms_scenario/sms_config.xml"),
            "--config-path", os.path.realpath("data/scenario/ile_de_france_config.xml"),
            "--count-links", os.path.realpath("data/daily_flow.csv"),
            "--config:linkStats.writeLinkStatsInterval", "0",
            "--config:controler.writeTripsInterval", "0",
            #"--use-vdf", "true",
        ]
    )
)

from boptx.tracker import PickleTracker
tracker = PickleTracker("optimization_{}.p".format(selected_objective))

from boptx.loop import Loop
Loop(
    algorithm = algorithm,
    evaluator = evaluator,
    maximum_evaluations = 4000,
).advance(callback = tracker)
