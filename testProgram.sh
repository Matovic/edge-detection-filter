#!/bin/bash

TIMEFORMAT=%R

# test on localhost
echo "processes,threads,time" > evaluation/time_tests.csv
host="127.0.0.1"
for numProcess in 1 2 3 4 8 16
do
#  echo ""
  for numThreads in 1 2 4 8 16 32 64 
  do
    hosts=""
    # get all hosts for 127.0.0.1
    for ((i=1; i <= $numProcess; ++i))
    do
	hosts+=$host
	if [ $i -ne $numProcess ]
	then
		hosts+=','
	fi;
    done
    #for lowThreshold in 1 10 50 100
    #do
    #  for ratio in 1 2 3
    #  do
        # -n gets rid off newline at the end
    echo -n "$numProcess,$numThreads," >> evaluation/time_tests.csv
    # time command displays the completion time of a script
    { time mpirun -n $numProcess --host $hosts _install/edge_detection --THREADS $numThreads --THRESHOLD 50 --RATIO 2 ; } 2>> evaluation/time_tests.csv
    #  done
    #done
  done
done
