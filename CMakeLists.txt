# basic struckture of cmakelist
cmake_minimum_required(VERSION 3.5)

# set compiler to be mpiCC, it better handles MPI
SET(CMAKE_CXX_COMPILER "/usr/bin/mpiCC.openmpi")

# set name of the project with CXX compiler
project(edge_detector CXX)

#
# CONFIGURATION
#

# Basic CMake settings
SET(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/_install)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
# Enable C++17
SET(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)

# MPI
find_package(MPI REQUIRED)
if(MPI_FOUND)
    message("Found MPI")
    message("Includes: " ${MPI_INCLUDE_DIRS})
endif(MPI_FOUND)

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("Found OpenMP")
    message("Includes: " ${OpenMP_INCLUDE_DIRS})
    
    # set C/C++ flags for OpenMP
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")	
    
    # set linker flags
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("Found OpenCV")
    message("Includes: " ${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

# edge detection
add_executable(edge_detection src/main.cpp)
target_link_libraries(edge_detection ${OpenCV_LIBS} ${MPI_LIBS})
install(TARGETS edge_detection DESTINATION .)
