# Edge Detection Filter

## Installation instructions

Depending on your system of choice you will need to install the following dependencies:
* [mpiCC](https://www.open-mpi.org/doc/current/man1/mpicc.1.php) - Open MPI C wrapper compiler 
* [CMake](https://cmake.org) build system
* [Open MPI](https://www.mpi-forum.org/) - The Message Passing Interface
* [OpenMP](https://www.openmp.org/) - The Open Multi-Processing
* [OpenCV](https://opencv.org/) - The Open Source Computer Vision Library

You can also use CMake directly to generate project files, see [Usage](#Usage).

## Usage

You can generate the project files by using the bash script from command line as shown below. It will generate the default for the given platform. It can be changed by adding a generator for your IDE. To find out all available generators, just run `cmake --help`.

```bash
./runCMake.sh src/main.cpp
```

After installation, the files should be installed into a new `_install` subdirectory. You can then run the examples as follows:

```bash
./runProgram.sh
```
