from boptx.evaluator import BaseEvaluator, DefaultEvaluation
from boptx.algorithms.opdyts import OpdytsProblem, OpdytsEvaluation
from boptx.problem import Problem, ContinuousParameter

import os, shutil, copy
import subprocess as sp
import multiprocessing

import pandas as pd
import numpy as np
import glob
import json, pickle, copy

import logging
logger = logging.getLogger(__name__)

class MATSimProblem(OpdytsProblem):
    def parameterize(self, settings, values, information):
        raise NotImplementedError()

    def process(self, output_path, values, information):
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

class TerminationTracker:
    def start(self, identifier, values, information, restart = None):
        raise NotImplementedError()

    def update(self, identifier, output_path):
        raise NotImplementedError()

    def has_terminated(self, identifier):
        raise NotImplementedError()

    def clean(self, identifier):
        raise NotImplementedError()

    def get_information(self, identifier):
        return None

class MATSimStatus:
    RUNNING = "running"
    ERROR = "error"
    DONE = "done"

def create_default_settings(input_settings, input_config = {}):
    settings = {
        "java_binary": "java",
        "arguments": [],
        "vm_arguments": [],
        "config": input_config
    }

    settings.update(input_settings)
    return settings

class MATSimEvaluator(BaseEvaluator):
    def __init__(self, problem: MATSimProblem, working_directory, parallelism = 1, config = {}, settings = {}, termination : TerminationTracker = None):
        super().__init__(parallelism = parallelism)

        if not os.path.exists(working_directory):
            raise RuntimeError("Working directory does not exist: %s" % working_directory)

        self.working_directory = working_directory
        self.problem = problem
        self.settings = create_default_settings(settings, config)
        self.termination = termination

        self.simulations = {}

    def start_evaluation(self, identifier, values, information):
        if identifier in self.simulations:
            raise RuntimeError("A simulation with identifier %s already exists." % identifier)

        # Enrich opdyts information
        information = copy.deepcopy(information)

        # Parametrize
        settings = copy.deepcopy(self.settings)
        self.problem.parameterize(settings, values, information)
        config = settings["config"]

        # Handle opdyts information
        first_iteration = None
        last_iteration = None
        restart_identifier = None

        if "opdyts" in information:
            if not "transition_size" in settings:
                raise RuntimeError("No transition size for odpyts is given")

            transition_size = settings["transition_size"]
            opdyts = information["opdyts"]

            if "restart" in opdyts:
                if "plans.inputPlansFile" in config:
                    raise RuntimeError("Need to override plans.inputPlansFile for opdyts, but value is already set")

                # Set input plans
                plans_path = self._get_output_file(opdyts["restart"], "output_plans.xml.gz")
                config["plans.inputPlansFile"] = os.path.realpath(plans_path)

                # Find the start iteration
                iterations = self._get_iterations(opdyts["restart"])
                first_iteration = iterations["current"]
                last_iteration = first_iteration + transition_size

                restart_identifier = opdyts["restart"]

            else: # We do not restart a simulation, so we only perform one iteration
                first_iteration = 0
                last_iteration = 0

        # Handle termination criterion
        if not self.termination is None and not "opdyts" in information:
            if not "transition_size" in settings:
                raise RuntimeError("The transition size must be given when using a termination criterion")

            first_iteration = 0
            last_iteration = settings["transition_size"]

            if "overwriteFiles" in config:
                raise RuntimeError("Attempting to set controler.overwriteExistingFiles for termination functionality, but value is already set")

            config["controler.overwriteFiles"] = "deleteDirectoryIfExists"

        # Handle iteration configuration
        if not first_iteration is None:
            if "controler.firstIteration" in config:
                raise RuntimeError("Attemting to override controler.firstIteration, but value is already set")

            config["controler.firstIteration"] = first_iteration

        if not last_iteration is None:
            if "controler.lastIteration" in config:
                raise RuntimeError("Attemting to override controler.lastIteration, but value is already set")

            config["controler.lastIteration"] = last_iteration

        # Output directory is standardized so we know where the files are
        if "controler.outputDirectory" in config:
            raise RuntimeError("Need to override controler.outputDirectory, but value is already set")

        simulation_path = "%s/%s" % (self.working_directory, identifier)
        config["controler.outputDirectory"] = "%s/output" % simulation_path

        if os.path.exists(simulation_path):
            shutil.rmtree(simulation_path)

        os.mkdir(simulation_path)

        # Create the process
        stdout = open("{}/simulation_output.log".format(simulation_path), "w+")
        stderr = open("{}/simulation_error.log".format(simulation_path), "w+")

        command_line = self._build_command_line(settings)

        logger.info("Starting simulation %s:" % identifier)
        logger.info(" ".join(command_line))

        process = sp.Popen(command_line, stdout = stdout, stderr = stderr)

        self.simulations[identifier] = {
            "process": process,
            "status": MATSimStatus.RUNNING,
            "values": values,
            "information": information,
            "settings": settings,
            "iteration": None,
            "transitional": False
        }

        if not self.termination is None:
            termination_restart = restart_identifier

            if "opdyts" in information:
                if "restart_convergence" in information["opdyts"]:
                    if not information["opdyts"]["restart_convergence"]:
                        termination_restart = None

            self.termination.start(identifier, values, information, restart = termination_restart)

    def _build_command_line(self, settings):
        # Construct command line
        command_line = [settings["java_binary"]]
        command_line += settings["vm_arguments"]

        if "memory" in settings:
            command_line += ["-Xmx{}".format(settings["memory"])]

        if not "class_path" in settings:
            raise RuntimeError("Parameter 'class_path' is missing")

        command_line += ["-cp", settings["class_path"]]

        if not "main_class" in settings:
            raise RuntimeError("Parameter 'main_class' is missing")

        command_line += [settings["main_class"]]
        command_line += settings["arguments"]

        for key, value in settings["config"].items():
            command_line += ["--config:{}".format(key), str(value)]

        return command_line

    def _advance(self, identifier):
        """ Advances a finished simulatio for one more transition """

        # Unpack references
        simulation_path = "%s/%s" % (self.working_directory, identifier)
        simulation = self.simulations[identifier]
        settings = simulation["settings"]

        # Handle input plans
        source_path = self._get_output_file(identifier, "output_plans.xml.gz")
        target_path = "{}/advance_plans.xml.gz".format(simulation_path)
        shutil.copy(source_path, target_path)

        settings["config"]["plans.inputPlansFile"] = os.path.realpath(target_path)
        settings["config"]["controler.overwriteFiles"] = "overwriteExistingFiles"

        # Define iterations
        iterations = self._get_iterations(identifier)

        first_iteration = iterations["current"]
        last_iteration = first_iteration + settings["transition_size"]

        settings["config"]["controler.firstIteration"] = first_iteration
        settings["config"]["controler.lastIteration"] = last_iteration

        # Create the process
        stdout = open("{}/simulation_output.log".format(simulation_path), "w+")
        stderr = open("{}/simulation_error.log".format(simulation_path), "w+")

        command_line = self._build_command_line(settings)

        logger.info("Advancing simulation %s:" % identifier)
        logger.info(" ".join(command_line))

        simulation["process"] = sp.Popen(command_line, stdout = stdout, stderr = stderr)

    def _get_output_file(self, identifier, suffix):
        candidates = glob.glob("{}/{}/output/*{}".format(self.working_directory, identifier, suffix))

        if len(candidates) == 1:
            if os.path.isfile(candidates[0]):
                return candidates[0]

        return None

    def _get_iterations(self, identifier):
        """ Obtain information on the first, last, and current iteration of a MATSim run """

        stopwatch_path = self._get_output_file(identifier, "stopwatch.txt")
        iterations = dict(first = 0, last = None, current = None)

        if not stopwatch_path is None:
            try:
                df = pd.read_csv(stopwatch_path, sep = "\t")

                if len(df) > 0:
                    iterations["current"] = df["Iteration"].max()
            except:
                pass

        config = self.simulations[identifier]["settings"]["config"]

        if "controler.firstIteration" in config:
            iterations["first"] = config["controler.firstIteration"]

        if "controler.lastIteration" in config:
            iterations["last"] = config["controler.lastIteration"]

        return iterations

    def _ping(self):
        """ Loop through running simulation to update their state """

        for identifier, simulation in self.simulations.items():
            if simulation["status"] == MATSimStatus.RUNNING:
                return_code = simulation["process"].poll()

                if return_code is None:
                    # MATSim is still running

                    progress = self._get_iterations(identifier)

                    if not progress["current"] is None and progress["current"] != simulation["iteration"]:
                        simulation["iteration"] = progress["current"]

                        logger.info("Running simulation {} ... ({}/{} iterations)".format(
                            identifier, progress["current"], "?" if progress["last"] is None else progress["last"]
                        ))

                elif return_code == 0:
                    # MATSim has finished successfully

                    has_finished = True

                    if not self.termination is None:
                        output_path = "{}/{}/output".format(self.working_directory, identifier)
                        self.termination.update(identifier, output_path)

                        information = self.termination.get_information(identifier)
                        if not information is None:
                            simulation["information"]["termination"] = information

                        has_terminated = self.termination.has_terminated(identifier)
                        simulation["transitional"] = not has_terminated

                        if not has_terminated and not "opdyts" in simulation["information"]:
                            has_finished = False
                            self._advance(identifier)

                    if has_finished:
                        logger.info("Finished simulation {}".format(identifier))
                        simulation["status"] = MATSimStatus.DONE

                else:
                    # MATSim has exited with an error
                    raise RuntimeError("Error running simulation {}. See {}/{}/simulation_error.log".format(
                        identifier, self.working_directory, identifier
                    ))

    def check_evaluation(self, identifier):
        self._ping()
        return self.simulations[identifier]["status"] == MATSimStatus.DONE

    def clean_evaluation(self, identifier):
        if not self.check_evaluation(identifier):
            raise RuntimeError("Simulation %s is not ready to obtain result." % identifier)

        del self.simulations[identifier]

        simulation_path = "%s/%s" % (self.working_directory, identifier)

        if os.path.exists(simulation_path):
            shutil.rmtree(simulation_path)

        if not self.termination is None:
            self.termination.clean(identifier)

    def get_evaluation(self, identifier):
        if not self.check_evaluation(identifier):
            raise RuntimeError("Simulation %s is not ready to obtain result." % identifier)

        simulation = self.simulations[identifier]
        output_path = "%s/%s/output" % (self.working_directory, identifier)

        values = simulation["values"]
        response = self.problem.process(output_path, simulation["values"], simulation["information"])

        information = simulation["information"]
        matsim = dict(iterations = self._get_iterations(identifier))
        information["matsim"] = matsim

        if "information" in response:
            matsim.update(response["information"])

        return MATSimEvaluation(
            values = simulation["values"],
            information = information,
            objective = response["objective"],
            state = response["state"] if "state" in response else [],
            transitional = simulation["transitional"]
        )

class ModeShareTracker(TerminationTracker):
    def __init__(self, modes, H, S, T):
        self.H = H
        self.S = S
        self.T = T
        self.modes = modes

        self.trajectories = {}
        self.terminated = {}

    def start(self, identifier, values, information, restart = None):
        print("TERMINATION: Start {} based on {}".format(identifier, restart))
        self.trajectories[identifier] = None
        self.terminated[identifier] = False

        if not restart is None:
            self.trajectories[identifier] = copy.deepcopy(self.trajectories[restart])

    def update(self, identifier, output_path):
        # Read mode shares
        candidates = glob.glob("{}/*modestats.txt".format(output_path))
        candidates = [c for c in candidates if not "ph_" in c]
        candidates = [c for c in candidates if not "pkm_" in c]
        assert len(candidates) == 1

        # Update trajectories
        df_new = pd.read_csv(candidates[0], sep = "\t")
        df_trajectory = self.trajectories[identifier]

        if df_trajectory is None:
            df_trajectory = df_new
        else:
            df_trajectory = pd.concat([
                df_trajectory.iloc[:-1], df_new
            ])

        self.trajectories[identifier] = df_trajectory

        # Analyze trajectories
        information = self.get_information(identifier)
        self.terminated[identifier] = information["terminated"]

        # Save temporary information
        with open("{}/boptx_termination.p".format(output_path), "wb+") as f:
            pickle.dump(information, f)

        # And a CSV for checking the current progress
        output = {}
        output["iteration"] = np.arange(len(information["all"]))

        for mode in self.modes:
            output["{}_y".format(mode)] = information["modes"][mode]["y"]
            output["{}_s".format(mode)] = information["modes"][mode]["s"]
            output["{}_d".format(mode)] = information["modes"][mode]["d"]
            output["{}_dm".format(mode)] = information["modes"][mode]["dm"]
            output["{}_dp".format(mode)] = information["modes"][mode]["dp"]
            output["{}_f".format(mode)] = information["modes"][mode]["f"]

        output["all_f"] = information["all"]
        output["T"] = information["settings"]["T"]

        pd.DataFrame(output).to_csv(
            "{}/boptx_termination.csv".format(output_path), sep = ";", index = None)

    def get_information(self, identifier):
        settings = { "H": self.H, "S": self.S, "T": self.T }
        information = { "modes": {}, "all": None, "terminated": False, "settings": settings }

        df_trajectory = self.trajectories[identifier]
        f_all = np.ones((len(df_trajectory),), dtype = bool)

        for mode in self.modes:
            information["modes"][mode] = {}

            y = df_trajectory[mode].values
            information["modes"][mode]["y"] = y

            s = np.ones((len(y),)) * np.nan
            s[self.S:-self.S] = [
                np.sum(y[k - self.S:k + self.S]) / (2 * self.S)
                for k in range(self.S, len(y) - self.S)]
            information["modes"][mode]["s"] = s

            d = np.ones((len(y),)) * np.nan
            d[self.H:-self.H] = [
                (s[k + self.H] - s[k - self.H]) / (2 * self.H)
                for k in range(self.H, len(y) - self.H)]
            information["modes"][mode]["d"] = d

            dm = np.ones((len(y),)) * np.nan
            dm[self.H:] = d[:-self.H]
            information["modes"][mode]["dm"] = dm

            dp = np.ones((len(y),)) * np.nan
            dp[:-self.H] = d[self.H:]
            information["modes"][mode]["dp"] = dp

            f = np.abs(dp) <= self.T
            f &= np.abs(dm) <= self.T
            f &= np.abs(d) <= self.T
            information["modes"][mode]["f"] = f

            f_all &= f

        information["all"] = f_all
        information["terminated"] = np.any(f_all)
        return information

    def has_terminated(self, identifier):
        return self.terminated[identifier]

    def clean(self, identifier):
        del self.trajectories[identifier]

class GlobalModeShareProblem(MATSimProblem):
    def __init__(self, reference, initial):
        self.reference = reference
        self.reference_modes = list(reference.keys())

        self.initial = initial
        self.parameter_modes = list(initial.keys())

    def get_parameters(self):
        return [
            ContinuousParameter(name = mode, initial_value = self.initial[mode])
            for mode in self.parameter_modes
        ]

    def get_state_count(self):
        return len(self.reference_modes)

    def parameterize(self, settings, values, information):
        for mode_index, mode in enumerate(self.parameter_modes):
            slot = "planCalcScore.scoringParameters[subpopulation=null].modeParams[mode={}].constant".format(mode)
            settings["config"][slot] = values[mode_index]

    def process(self, output_path, values, information):
        candidates = glob.glob("{}/*modestats.txt".format(output_path))
        candidates = [c for c in candidates if not "ph_" in c]
        candidates = [c for c in candidates if not "pkm_" in c]
        assert len(candidates) == 1

        df = pd.read_csv(candidates[0], sep = "\t")

        state = np.array([
            df[mode].values[-1] - self.reference[mode]
            for mode in self.reference_modes
        ])

        objective = np.sum(state**2)

        return dict(
            objective = objective,
            state = state
        )
