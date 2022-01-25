# Setting up the environment

The calibration framework is packaged as a Python software. For dependency
management, we make use of [Anaconda](https://www.anaconda.com/). As part of
the repository, we provide `environment.yml` which describes exactly the packages
and versions needed to use the framework.

## Setting up the environment (GUI)

To set up the environment, you can use the GUI that is provided by Anaconda. Just
create a new environment using the GUI and provide the path to `environment.yml`.
Whenever you want to use the calibration framework, make use to start a new
shell session from the Anaconda GUI.

## Setting up the environment (command line)

If you use Anaconda on the command line, simply create a new environment called
`boptx` by executing:

```bash
conda env create -f environment.yml -n boptx
```

Make sure to always enter the environment before you use the framework:

```bash
conda activate boptx
```

## Setting up the code

As our calibration framework is structured as a reuseable Python package, it
needs to be registered as a development package in the created environment. To
do so, run once after creating your environment:

```bash
conda develop src
```

The command needs to be executed in the main directory of the repository (the
one containing the `src` folder).

## Testing the environment

At this point, you should be good to go. You can test that everything works
smoothly by running the unit tests of the framework:

```bash
python3 -m pytest tests
```

## Run your first calibration

Have a look at the [introductory example](../examples/simple) to run your first calibration.
