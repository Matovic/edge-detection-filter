# Edge Detection Filter

Programs outputs, images with edge detection, are located in a new `output` subdirectory based on input images found in the `data` folder. 

## Installation instructions

Depending on your system of choice you will need to install the following dependencies:
* mpiCC - Open MPI C++ wrapper compiler 
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

After installation, the file should be installed into a new `_install` subdirectory. You can then run tests as follows:

```bash
./testProgram.sh
```
## Evaluation

In the evaluation folder, we provided time tests. We created a bash script for the time tests that write a specific time of each program's execution in a CSV file. Subsequently, we conducted an evaluation using the programming language R, where we evaluated our tests using descriptive statistics.

Preview from the evaluation file:  

<p align="center">
	<img src="./figures/time_tests.png">
</p>
