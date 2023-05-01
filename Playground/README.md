# Out-of-place PGM

## Development Environment

The dockerfile can be built by running

`make build`

from inside the `Playground` directory. There are a handful of other helpful `make` commands (from the `Playground` directory) to create a container for this image and use it for testing.

- `make startcontainer` - Starts a container pointing to the `playground` image generated above. The container `id` is stored in a local file.

- `make shell` - Creates an interactive shell inside the currently active container.

- `make stopcontainer` - Stops the current container.

## Overview of Files

### `src/include`

- `buffered` folder
  - In keeping with the style of the original index, the `BufferedPGMIndex` (which admits out-of-place writes into leaf segment buffers) has a header-only implementation in `src/included/buffered/pgm_index_buffered.hpp`. It's worth noting that while the `BufferedPGMIndex` is contained in a different folder, it is included as part of the `pgm` namespace.
- `debug` folder
  - Header-only implementation of useful index metrics in `metrics.hpp` and a progressbar in `progressbar.hpp`.
- `pgm` folder
  - Implementations of all the standard PGM variants from the original paper

### `src/experiments`

- `experiments.h` - Header file including definitions of all shared experiment functions and experiment implementations.
- `common.cpp` - Helper functions used across various experiments. Examples include helper functions to generate data, time reads, perform inserts, etc.
- `TEMPLATE.cpp` - A template for what experiments should look like. Each experiment should have a unique short name, and clearly define it's goals, inputs, and outputs at the top of the file.
- `[experiment_name].cpp` - A specific experiment. You can investigate the comment at the top of the file to get more detailed in

Important notes about experiments:

- Each experiment should take in an output filename. This results will be written to `src/build/[filename]`.
- Most results so far are simply reported as CSVs with named columns.
- The metrics contained in `metrics.cpp` have `encode` functions which puts them into a string representation, which can be parsed by functions from `results/metrics/schema.py`.
- Getting specific results is relatively hands-on. It is on you to explicitly define the format of the output file in the experiment and write results when appropriate.
  - NOTE: This is something I'm actively working on changing, so that one can run experiments that automatically capture results after only specifying a workload and the configurations to test.
- To get a feel for how to compare various configurations, take a look at an existing experiment, specifically the `Configuration` struct.
  - NOTE: I'm also working on moving this `Configuration` struct out of specific files so it is shared across experiments.

### `src/build`

Just contains auto-generated build files and the like. The most important thing to note is that if you specify an output file in an experiment, it will be written here (unless explicitly stated otherwise). For the time being I've just been copying these files to dedicated `results` folder as needed.

### `src/runner.cpp`

The experiment runner. A bit tedious, but import the experiment you want to run and call it with a sensibly named filename.

## Make Commands (from `src`)

- `make compile` - Compiles the entire project.
- `make runner` - Compiles and runs the entire project (will end up running whatever is in `runner.cpp`).
- `make valgrind-full` - Same as `make runner` but with valgrind memory profiling.
