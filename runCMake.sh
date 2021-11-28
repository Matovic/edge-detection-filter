#!/bin/bash

# error message function
error_msg() {
	# msg="Error: 'Invalid input, try $0 -h for help.'"
	echo -e $1 >&2 						# redirect stdout to stderr
	exit 1							# error exit
}

EXPECTED_ARGS=1
dir=$(pwd)'/'

# check number of arg
if [ $# -ne $EXPECTED_ARGS ]
then
	error_msg  "\nInvalid number of arguments, please check the inputs and try again.\n\nExpected name of C/C++ file\n"

# check if given file exist
elif ! [ -f "${dir}$1" ]
then
	error_msg "Error: 'File $_file does not exist.'"	

else
        echo -e "Running cmake"
	if [ -d "${dir}_build" ] 
	then
		rm -r "${dir}_build"
	fi
	#mpiCC $1 -fopenmp -o output.out `pkg-config --cflags --libs opencv4`
	mkdir _build && cd _build
	cmake ..
	cmake --build . --target install #--config Release #--target install
	cd ..
        echo -e "\nDone!\n"

fi;
