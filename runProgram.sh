#!/bin/bash

TIMEFORMAT=%R
#for numTest in 1 2 3
#do
  echo ""
  for numThreads in 1 2 4 8 16 32 64 
  do
    echo -n "[OpenMP] $numThreads thread" # ($numTest. test): "
    time ./_install/edge_detection --THREADS $numThreads		# time command displays the completion time of a script
    #echo -n "[MPI] $n thread ($l. test): "
    #time mpirun -n $n ./p2mpi
  done
#done
