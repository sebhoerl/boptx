# boptx

This repository contains a framework for calibration large-scale agent-based
transport simulations. It is mainly developed to calibrate simulations from the
MATSim-based eqasim framework, which offers standardized simulation cases for
various cities in France, such as Paris, Lyon, Nantes, or Toulouse. However, it
can be adapted to any other simulator. The framework follows three main goals:

- Provide tested implementations of common optimization and calibration approaches,
most of them being black-box optimization algorithms, and some of them implementing
specific aspects to speed up the calibration of transport simulations.
- Provide a toolset to quickly set up calibration pipelines for common data sources
and calibration objectives in transport planning, such as traffic flows, mode shares,
or travel times.
- Provide wrappers for common transport planning simulations and models.

Have a look at the [list of implemented algorithms](docs/Algorithms.md), among others including implementations from elitist and non-elitist
evolutionary search strategies such as **CMA-ES** and **xNES** to gradient approximation
methods such as **FDSA** and **SPSA**.

To get started with the framework, please have a look at our instructions on how to prepare the environment and read the introductory example:

- **[Setting up the environment](docs/Environment.md)**
- **[Introductory example](examples/simple)**

## Things to do

> Attention! The first use cases of the framework have indicated a couple of improvements that can be applied to the API. These will be implemented over the coming months. In case you're already using the framework, make sure to base your developments on a fixed commit to avoid any inconsistencies along the way.

This repository has just been put online. Here is a list of things that still need
to be done:

- [ ] We already have unit tests, but not yet in the repository. Put them in.
- [ ] We already have code for Bayesian Optimization. Cleanup and put it in.
- [ ] Remove TODOs in simple examples
- [ ] Write the example for eqasim Paris as desribed by TODO
- [ ] Write the example for a standard MATSim simulation
- [ ] Write the example for a standard SUMO simulation
- [ ] Write the page explaining all the implemented algorithms (switch to some automatic documentation format?)

## Examples

Currently, three examples are provided in this repository:

- **[Simple](examples/simple)**: A simple example that describes how to wrap up any simulator for the calibration framework. The process is demonstrated with a simple multi-nomial discrete choice model which, non-linearly, depends on the resulting mode share. It is used to benchmark different algorithms. There is also a **[simple opdyts](examples/simple_opdyts)** example for the *opdyts* search acceleration startegy by [Flötteröd (2017)](https://www.sciencedirect.com/science/article/pii/S0191261516302466).
- **[MATSim Berlin](examples/matsim)**: An example using [MATSim](https://matsim.org/) that makes use of the [Open Berlin Scenario](https://github.com/matsim-scenarios/matsim-berlin). It demonstrates how to calibrate mode-specific constants of a MATSim simulation to fit overall mode shares, inspired by the abstract by [Agarwal et al.](https://transp-or.epfl.ch/heart/2017/abstracts/hEART2017_paper_109.pdf). Additionally, an example for opdyts is given.
- **[Eqasim Île-de-France](examples/eqasim)**: An example for [eqasim](https://github.com/eqasim-org/eqasim-java), which describes how to calibrate mode shares (by distance class), traffic flows, and travel times for [an open scenario of Paris and Île-de-France](examples/eqasim) that can be adapted to Lyon, Nantes, and other French cities.
- **[SUMO Monaco](examples/sumo)**: An example for [SUMO](examples/sumo), where we calibrate network parameters in an [open scenario for Monaco](https://github.com/lcodeca/MoSTScenario).
