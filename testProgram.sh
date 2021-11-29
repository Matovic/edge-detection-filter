#!/bin/bash

TIMEFORMAT=%R

# 
#echo "processes,threads,time" > evaluation/time_tests.csv
#for numProcess in 1 2 3 4 8 16
#do
#  echo ""
#  for numThreads in 1 2 4 8 16 32 64 
#  do
#    # -n gets rid off newline at the end
#    echo -n "$numProcess,$numThreads," >> evaluation/time_tests.csv
#    # time command displays the completion time of a script
#    { time mpirun -oversubscribe -n $numProcess --host 127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1 _install/edge_detection --THREADS $numThreads ; } 2>> evaluation/time_tests.csv
#  done
#done

for lowThreshold in 1 10 50 100
do
  for ratio in 1 2 3 4 5
  do
  	./_install/edge_detection --THRESHOLD $lowThreshold --RATIO $ratio
  done
done
