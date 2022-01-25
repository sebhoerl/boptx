from boptx.evaluator import BaseEvaluator, DefaultEvaluation
from boptx.algorithms.opdyts import OpdytsProblem, OpdytsEvaluation
from boptx.problem import Problem

import os, shutil
import subprocess as sp
import pandas as pd
import numpy as np
import glob
import json

import logging
logger = logging.getLogger(__name__)

class MATSimProblem(Problem):
    def parameterize(self, values):
        raise NotImplementedError()

    def evaluate(self, values, path):
        raise NotImplementedError()

    def get_state_count(self):
        raise NotImplementedError()

class MATSimEvaluation(DefaultEvaluation, OpdytsEvaluation):
    def __init__(self, values, objective, information, state, transitional):
        super().__init__(values, objective, information)
        self.state = state
        self.transitional = transitional

    def get_state(self):
        return self.state

    def is_transitional(self):
        return self.transitional

class MATSimEvaluator(BaseEvaluator):
    """
        Defines a wrapper around a standard MATSim simulation
    """

    def __init__(self, working_directory, problem : MATSimProblem, settings = {}, parallelism = 1):
        super().__init__(parallelism = parallelism)

        if not os.path.exists(working_directory):
            raise RuntimeError("Working directory does not exist: %s" % working_directory)

        self.working_directory = os.path.realpath(working_directory)
        self.settings = settings

        if not "memory" in self.settings:
            self.settings["memory"] = "10G"

        if not "java" in self.settings:
            self.settings["java"] = "java"

        if not "threads" in self.settings:
            self.settings["threads"] = None

        if not "arguments" in self.settings:
            self.settings["arguments"] = []

        if not "config" in self.settings:
            self.settings["config"] = {}

        self.simulations = {}
        self.problem = problem

    def start_evaluation(self, identifier, values, information):
        response = self.problem.parameterize(values)

        if "type" in information:
            response["type"] = information["type"]

        if "restart" in information:
            response["restart"] = information["restart"]

        # TODO: HACK
        if "type" in response and response["type"] == OpdytsEvaluation.INITIAL:
            response["iterations"] = 400

        # Shortcut for invalid cases
        if "skip" in response and response["skip"]:
            self.simulations[identifier] = {
                "status": "done",
                "values": values,
                "iterations": None, "arguments": [],
                "information": information
            }

            logger.info("Skipping simulation {}".format(identifier))
            return

        if identifier in self.simulations:
            raise RuntimeError("A simulation with identifier %s already exists." % identifier)

        # Prepare the working space
        simulation_path = "%s/%s" % (self.working_directory, identifier)

        if os.path.exists(simulation_path):
            shutil.rmtree(simulation_path)

        os.mkdir(simulation_path)

        # Merge config
        config = dict()
        config.update(self.settings["config"])
        config.update(response["config"])

        # Merge arguments
        arguments = []
        arguments += self.settings["arguments"]
        arguments += response["arguments"]

        # A specific random seed is requested
        if "random_seed" in response or "random_seed" in self.settings:
            random_seed = None
            if "random_seed" in self.settings: random_seed = self.settings["random_seed"]
            if "random_seed" in response: random_seed = response["random_seed"]

            if "global.random_seed" in config:
                logger.warn("Overwriting 'global.random_seed' for simulation %s" % identifier)

            config["global.random_seed"] = random_seed

        # It is requested to restart from a certain existing simulation output
        if "restart" in response and "type" in response:
            if "plans.inputPlansFile" in config:
                logger.warn("Overwriting 'plans.inputPlansFile' for simulation %s" % identifier)

            if "controler.firstIteration" in config:
                logger.warn("Overwriting 'controler.firstIteration' for simulation %s" % identifier)

            restart_path = "%s/%s" % (self.working_directory, response["restart"])

            # Find the iteration at which to start
            df_stopwatch = pd.read_csv("%s/output/stopwatch.txt" % restart_path, sep = "\t")
            first_iteration = df_stopwatch["Iteration"].max()

            config["plans.inputPlansFile"] = "%s/output/output_plans.xml.gz" % restart_path

            ignore_convergence = response["type"] == OpdytsEvaluation.CANDIDATE

            if not ignore_convergence:
                config["controler.firstIteration"] = first_iteration
                arguments += ["--signal-input-path", restart_path + "/output"]

        # A certain number of iterations is requested
        iterations = None

        if "iterations" in response or "iterations" in self.settings:
            if "iterations" in self.settings: iterations = self.settings["iterations"]
            if "iterations" in response: iterations = response["iterations"]

            if "controler.lastIteration" in config:
                logger.warn("Overwriting 'controler.lastIteration' for simulation %s" % identifier)

            if "controler.writeEventsInterval" in config:
                logger.warn("Overwriting 'controler.writeEventsInterval' for simulation %s" % identifier)

            if "controler.writePlansInterval" in config:
                logger.warn("Overwriting 'controler.writePlansInterval' for simulation %s" % identifier)

            last_iteration = iterations

            # In case firstIteration is set, we need to add the number here
            if "controler.firstIteration" in config:
                last_iteration += config["controler.firstIteration"]

            config["controler.lastIteration"] = last_iteration
            config["controler.writeEventsInterval"] = last_iteration
            config["controler.writePlansInterval"] = last_iteration

        # Output directory is standardized so we know where the files are
        if "controler.outputDirectory" in config:
            logger.warn("Overwriting 'controler.outputDirectory' for simulation %s" % identifier)

        config["controler.outputDirectory"] = "%s/output" % simulation_path

        # Construct command line arguments
        if not "class_path" in response and not "class_path" in self.settings:
            raise RuntimeError("Parameter 'class_path' must be set for the MATSim simulator.")

        if not "main_class" in response and not "main_class" in self.settings:
            raise RuntimeError("Parameter 'main_class' must be set for the MATSim simulator.")

        class_path = None
        if "class_path" in self.settings: class_path = self.settings["class_path"]
        if "class_path" in response: class_path = response["class_path"]

        main_class = None
        if "main_class" in self.settings: main_class = self.settings["main_class"]
        if "main_class" in response: main_class = response["main_class"]

        java = self.settings["java"]
        if "java" in response: java = response["java"]

        memory = self.settings["memory"]
        if "memory" in response: memory = response["memory"]

        threads = self.settings["threads"]
        if "threads" in response: threads = response["threads"]

        if not threads is None:
            if "global.numberOfThreads" in config:
                logger.warn("Overwriting 'global.numberOfThreads' for simulation %s" % identifier)

            config["global.numberOfThreads"] = threads

            if "qsim.numberOfThreads" in config:
                logger.warn("Overwriting 'qsim.numberOfThreads' for simulation %s" % identifier)

            config["qsim.numberOfThreads"] = min(12, threads)

        arguments = [
            java, "-Xmx%s" % memory,
            "-cp", class_path, main_class
        ] + arguments

        for key, value in config.items():
            arguments += ["--config:%s" % key, str(value)]

        arguments = [str(a) for a in arguments]

        stdout = open("%s/simulation_output.log" % simulation_path, "w+")
        stderr = open("%s/simulation_error.log" % simulation_path, "w+")

        logger.info("Starting simulation %s:" % identifier)
        logger.info(" ".join(arguments))

        self.simulations[identifier] = {
            "process": sp.Popen(arguments, stdout = stdout, stderr = stderr),
            "arguments": arguments, "status": "running", "progress": -1,
            "iterations": iterations,
            "values": values,
            "information": information
        }

    def _ping(self):
        for identifier, simulation in self.simulations.items():
            if simulation["status"] == "running":
                return_code = simulation["process"].poll()

                if return_code is None:
                    # Still running!
                    iteration = self._get_iteration(identifier)

                    if iteration > simulation["progress"]:
                        simulation["progress"] = iteration

                        logger.info("Running simulation {} ... ({}/{} iterations)".format(
                            identifier, iteration, "?" if simulation["iterations"] is None else simulation["iterations"]
                        ))

                elif return_code == 0:
                    # Finished
                    logger.info("Finished simulation {}".format(identifier))
                    simulation["status"] = "done"
                else:
                    # Errorerd
                    del self.simulations[identifier]
                    raise RuntimeError("Error running simulation {}. See {}/{}/simulation_error.log".format(identifier, self.working_directory, identifier))

    def _get_iteration(self, identifier):
        stopwatch_paths = glob.glob("%s/%s/output/*stopwatch.txt" % (self.working_directory, identifier))

        if len(stopwatch_paths) > 0 and os.path.isfile(stopwatch_paths[0]):
            try:
                df = pd.read_csv(stopwatch_paths[0], sep = "\t")

                if len(df) > 0:
                    return df["Iteration"].max()
            except:
                pass

        return -1

    def check_evaluation(self, identifier):
        self._ping()
        return self.simulations[identifier]["status"] == "done"

    def get_evaluation(self, identifier):
        if not self.check_evaluation(identifier):
            raise RuntimeError("Simulation %s is not ready to obtain result." % identifier)

        simulation = self.simulations[identifier]
        output_path = "%s/%s/output" % (self.working_directory, identifier)
        objective_value, objective_information = self.problem.evaluate(simulation["values"], output_path)
        transitional = False

        if os.path.exists("%s/convergence.csv" % output_path):
            df_convergence = pd.read_csv("%s/convergence.csv" % output_path, sep = ";")
            transitional = df_convergence["active"].values[-1] > 0

        information = simulation["information"]
        information["matsim"] = {
           "iterations": self._get_iteration(identifier),
           "arguments": simulation["arguments"],
           "objective": objective_information,
           "identifier": identifier
        }

        return MATSimEvaluation(
             simulation["values"], objective_value, information,
             objective_information["states"], transitional
        )

    def clean_evaluation(self, identifier):
        del self.simulations[identifier]

        simulation_path = "%s/%s" % (self.working_directory, identifier)

        if os.path.exists(simulation_path):
            shutil.rmtree(simulation_path)
