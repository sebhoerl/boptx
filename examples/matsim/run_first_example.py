from boptx.matsim import MATSimEvaluator, MATSimProblem
from boptx.problem import ContinuousParameter
import time, os, shutil
import logging
logging.basicConfig(level = logging.INFO)

# git clone https://github.com/matsim-scenarios/matsim-berlin
# ./mvnw clean package -DskipTests=true

# Define the problem
class CustomProblem(MATSimProblem):
    def __init__(self):
        pass

    def parameterize(self, settings, values, information):
        print("Parametrizing simulation:", values)

        settings["config"]["controler.lastIteration"] = 10
        settings["config"]["planCalcScore.scoringParameters[subpopulation=null].modeParams[mode=car].constant"] = values[0]

    def process(self, output_path, values, information):
        print("Processing finished simulation:", output_path)
        return dict(objective = 0.0)

# Clean up working directory
if os.path.exists("working_directory"):
    shutil.rmtree("working_directory")
os.mkdir("working_directory")

evaluator = MATSimEvaluator(
    problem = CustomProblem(),
    working_directory = "working_directory",
    settings = dict(
        class_path = "matsim-berlin/matsim-berlin-5.5.3.jar",
        main_class = "org.matsim.run.RunBerlinScenario",
        #arguments = ["matsim-berlin/scenarios/berlin-v5.5-1pct/input/berlin-v5.5-1pct.config.xml"],
        arguments = ["matsim-berlin/scenarios/equil/config.xml"],
        memory = "8g"
    )
)

# Start an evaluation
evaluator.start_evaluation("my_run", [0.0], {})

# Wait until it is finished
while not evaluator.check_evaluation("my_run"):
    time.sleep(1)

# Obtain the result
evaluator.get_evaluation("my_run")

# Clean the evaluation
evaluator.clean_evaluation("my_run")