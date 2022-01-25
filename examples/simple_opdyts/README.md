# Simple opdyts example

This example is based on the [introductory example](../simple), but uses the
odpyts search acceleration approach by [Flötteröd (2017)](https://www.sciencedirect.com/science/article/pii/S0191261516302466). The basic logic of using opdyts in
combination with general evolutionary search algorithm has been explained in
TODO. The model is the same as in the introdutory example.

## Basic idea

The odpyts startegy wraps around another optimization algorithm (here we use
elitist CMA-ES as in the introductory example) and requests `N` parameter value
samples. It then advances each respective simulation by `T` (internal) iterations.
Hence, the idea is here to advance the simple mode choice equilibrium model by
only `T` iterations and then examine the transitional result. Then, based on the
`N` transitional states, odpyts decides step by step which simulation to advance
further by `T` steps. Hence, it is able to early filter out runs that are not
promising. The simulation run that finishes first (i.e. the model has converged)
becomes the new best candidate.

## Partial model runs

The introductory model already provides the functionality to provide an initial
state of a run and also to define a fixed number of iterations. The model will
then run until either this fixed number is reached or the convergence criterion
becomes active.

You can use [Model.ipynb](Model.ipynb) to play around with running the model
partially. In the notebook, four chunks of 100 iterations are run and then
plotted:

TODO

## Calibration problem

When using opdyts, the calibration problem becomes more complex. The three main
elements are:

- Opdyts expects an `OpdytsEvaluation` as output of the evaluator, which provides
additional information compared to the `DefaultEvaulation` which is used in the
introductory exaple. Specifically, opdyts is interested in the `state`
(for which we use the current mode shares) and whether the simulation run is converged
or not (whether it is *transitional*).
- Opdyts expects a `OpdytsProblem`, which also provides the number of states (here 3)
on top of the default information in `ContinuousProblem`.
- The `ModeShareEvaluator` needs to be adjusted such that it allows restarting
existing runs and advancing them by `T` iterations.

To resonse to the first two requirements, we define a `ModeShareEvaluation`,
which extends `DefaultEvaulation` and `OpdytsEvaluation`, and a `ModeShareProblem`,
which extends `ContinuousProblem` and `OpdytsProblem`. The implementations are
relatively straight-forward and can be found in [wrapper.py](wrapper.py).

The modifications for the evaluator are more advanced. For every request of an
evaluation, we not only save the parameter values and simulation results, but also
whether the simulation has converged.

When starting a simulation, we check whether opdyts has sent us `restart` information
via the `information` object passed to `start_evaluation`. If so, we use as the initial
state of the new simulation the last that of the one we should restart:

```python
initial_state = None

if "restart" in information:
    # Get the last state of the simulation we should restart
    initial_state = self.results[information["restart"]][-1]
```

We then run the model only for a limited number of iterations defined
by `transition_size`:

```python
self.results[identifier] = model.run(
    iterations = self.transition_size,
    initial_state = initial_state)
```

We find out whether the simulation has converged by checking whether the
model was running less iterations than requested (i.e. whether we ran into
the convergence criterion):

```python
self.converged[identifier] = len(self.results[identifier]) < self.transition_size
```

These information are then passed to the simulator in `get_evaluation` via
an `ModeShareEvaluation` object. Here, we also pass the current state:

```python
return ModeShareEvaluation(
    self.values[identifier], objective, self.information[identifier],
    state, not self.converged[identifier]
)
```

## Performing the calibration

The calibration procedure is very similar to the introductory example and can
be found in [Calibration.ipynb](Calibration.ipynb). A major change is the
definition of the algorithm:

```python
algorithm = CMAES1P1Algorithm(problem)
algorithm = OpdytsAlgorithm(problem, algorithm)
```

Here, we define CMA-(1,1)-ES as the sampling algorithm and wrap opdyts around it.
The notebook runs the loop for 3000 evaluations. Note that from the perspective
of the framework, one evaluation is now a transition of 20 model iterations, that
is why a much higher number is chosen than in the introductory example.

The notebook produces the following output:

TODO

The top row shows the same plot as in the introductory example. On the left, a
clear pattern of an opdyts calibration can be see: As simulations are advanced
transition by transition, we see how the objective values often converge slowly
to a final value in a series of points, unless we have switching behavior between
multiple "parallel" runs. Most points respresent, hence, transitional states
of the simulation.

The bottom line in the figure shows only those objective and parameter values
with `Evaluation`s that are not transitional. We see similar dynamics as in the
introductory example. One could now create a direct comparison in terms of
*model iterations* to have a fair comparison in how many computational steps
have been necessary to arrive at a certain objective value.
