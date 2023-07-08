# Limousine Out-Of-Place Writes

## Development Environment

The dockerfile can be built by running

`make build`

from inside the root directory. There are a handful of other helpful `make` commands to create a container for this image and use it for testing.

- `make startcontainer` - Starts a container pointing to the `oop-pgm` image generated above. The container `id` is stored in a local file.

- `make shell` - Creates an interactive shell inside the currently active container.

- `make stopcontainer` - Stops the current container.

## Overview of Files

### `src/include`

- `oop` folder
  - In keeping with the style of the original index, the `OopPGMIndex` (which admits out-of-place writes into leaf segment buffers) has a header-only implementation in `src/included/oop/pgm_index_buffered.hpp`. It's worth noting that while the `OopPGMIndex` is contained in a different folder, it is included as part of the `pgm` namespace.
- `debug` folder
  - Header-only implementation of useful index metrics in `metrics.hpp` and a progressbar in `progressbar.hpp`.
- `pgm` folder
  - Implementations of all the standard PGM variants from the original paper

### `src/experiments`

- `experiments.h` - Header file including definitions of all shared experiment functions and experiment implementations.
- `common.cpp` - Helper functions used across various experiments. Examples include helper functions to generate data, time reads, perform inserts, etc.
- `[experiment_name].cpp` - A specific experiment. Each experiment has a more detailed description at the top of the file which breaks down its aim, input, and output.

Important notes about experiments:

- Each experiment should take in an output filename. This results will be written to `src/build/[filename]`.
- Most results are exported as simple CSV files.
- The metrics contained in `metrics.cpp` have `encode` functions which puts them into a string representation, which can be parsed during analysis.
- To get a feel for how to compare various configurations, take a look at an existing experiment, specifically the `Configuration` struct.

### `src/runner.cpp`

The experiment runner. Simply imports the experiment to run and runs it from `main`.

## Make Commands (from `src`)

- `make compile` - Compiles the entire project.
- `make runner` - Compiles and runs the entire project (will end up running whatever is in `runner.cpp`).
- `make valgrind-full` - Same as `make runner` but with valgrind memory profiling.
