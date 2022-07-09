from boptx.matsim import MATSimEvaluator, MATSimProblem, ModeShareTracker, GlobalModeShareProblem
from boptx.problem import ContinuousParameter
from boptx.algorithms import CMAESAlgorithm, OpdytsAlgorithm
import time, os, shutil
import logging
logging.basicConfig(level = logging.DEBUG, datefmt = "%Y-%m-%d %H:%M:%S")

# git clone https://github.com/matsim-scenarios/matsim-berlin
# ./mvnw clean package -DskipTests=true

# Define the problem
problem = GlobalModeShareProblem(
    reference = { "car": 0.31, "pt": 0.18, "bicycle": 0.18, "walk": 0.22 },
    initial = { "car": 0.0, "pt": 0.0, "bicycle": 0.0 }
)

# Define the termination tracker
termination = ModeShareTracker(
    modes = ["car", "pt", "bicycle", "walk"],
    H = 50, S = 40, T = 1e-05
)

# Clean up working directory
if os.path.exists("working_directory"):
    shutil.rmtree("working_directory")
os.mkdir("working_directory")

# Define the evaluator
evaluator = MATSimEvaluator(
    problem = problem,
    termination = termination,
    working_directory = "working_directory",
    settings = dict(
        class_path = "matsim-berlin/matsim-berlin-5.5.3.jar",
        main_class = "org.matsim.run.RunBerlinScenario",
        vm_arguments = ["-Dmatsim.preferLocalDtds=true"],
        memory = "20g",
        arguments = ["matsim-berlin/scenarios/berlin-v5.5-1pct/input/berlin-v5.5-1pct.config.xml"],
        transition_size = 50,
        config = {
            "global.numberOfThreads": 8,
            "qsim.numberOfThreads": 8,
            "strategy.fractionOfIterationsToDisableInnovation": 9999.0
        }
    )
)

# Define the algorithm
algorithm = CMAESAlgorithm(problem)
algorithm = OpdytsAlgorithm(problem, algorithm)

# Prepare the calibration loop
from boptx.loop import Loop

loop = Loop(
    algorithm = algorithm,
    evaluator = evaluator,
    maximum_evaluations = 100,
)

# Prepare tracking of the calibration
from boptx.tracker import PickleTracker
loop.advance(callback = PickleTracker("calibration_opdyts.p"))
