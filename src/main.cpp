/**
* Course: Parallel Programming (FIIT - WS 2021/2022)
* Purpose:
*
* @file main.cpp
* @author Erik Matovic
* @version 30.11.2021
*/

#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

// Checks if string is an integer number.
bool isStringInt(const std::string& str)
{
    for (auto it = str.begin(); it != str.end(); ++it)
        if (std::isdigit(*it) == 0)
            return false;
    return !str.empty();
}

// Checks if given command line argument is correct number for an Othello game.
int checkNumberOnConsoleSwitch(const std::string consoleSwitch)
{
    if (!isStringInt(consoleSwitch))
        return -2;
    return std::stoi(consoleSwitch);
}

// Makes all characters uppercase in given string.
void toUpper(std::string& str)
{
    for (char& c : str)
        c = static_cast<char>(std::toupper(c));
}

// Processes command line arguments into a tuple of number of threads, max depth, heuristic function and time.
int processArguments(const int& argc, char* argv[])
{
    int numThreads = -1;    // numThreads = -1 is not valid
    std::vector<std::string> args(argv + 1, argc + argv);

    for (std::string consoleSwitch : args)
    {
        // make all characters uppercase in given string
        toUpper(consoleSwitch);

        // check given numbers on command line & set up if it is right
        if (numThreads == 0)
            numThreads = checkNumberOnConsoleSwitch(consoleSwitch);

        // check given command on command line is --THREADS & numThreads is not set up
        else if (consoleSwitch.compare("--THREADS") == 0 && numThreads == -1)
            numThreads = 0;

        // given command is wrong
        else
        {
            std::cerr << "ERROR 00: Wrong parameters!\n";
            std::exit(EXIT_FAILURE);
        }
    }
    return numThreads;
}

// Checks if given command line arguments are fit to be parameters for an Othello game.
int checkParameters(const int& numThreads)
{
    if (numThreads < 1)
    {
        std::cerr << "ERROR 01: Number of threads can not be less than 1!\n";
        return 1;
    }
    return 0;
}

// Get parameters from command line arguments.
int getParameters(const int& argc, char* argv[])
{
    // numThreads = -1, maxDepth = -1, heuristic = -1, moveTime = -1 are all not valid
    int numThreads = processArguments(argc, argv);
    //numThreads = processArguments(argc, argv);

    if (checkParameters(numThreads))
        std::exit(EXIT_FAILURE);

    return numThreads;
}

int main(int argc, char *argv[]) 
{
    // argc == 1 is without flags
	if (argc != 3)          // expected --THREADS <number of threads>
	{
		std::cerr << "ERROR 00: Invalid parameters\n";
		return 1;
	}

    // numThreads = -1, maxDepth = -1, heuristic = -1, moveTime = -1 are all not valid
    int numThreads = -1;
    numThreads = getParameters(argc, argv);
/*
	// Reading image
	cv::Mat img = cv::imread("data/4987_21_HE.tif");
	// Set window
	cv::namedWindow("Display frame", cv::WINDOW_NORMAL);
	//Resize window
	cv::resizeWindow("Display frame", 512, 512);
	// Display original image
	cv::imshow("Display frame", img);
	cv::waitKey(0);
	
	// Convert to graycsale
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	
	// Blur the image for better edge detection
	cv::Mat img_blur;
	cv::GaussianBlur(img_gray, img_blur, cv::Size(3,3), 0);

	// Sobel edge detection
	cv::Mat sobelx, sobely, sobelxy;
	cv::Sobel(img_blur, sobelx, CV_64F, 1, 0, 5);
	cv::Sobel(img_blur, sobely, CV_64F, 0, 1, 5);
	cv::Sobel(img_blur, sobelxy, CV_64F, 1, 1, 5);
	
	// Display Sobel edge detection images
	cv::imshow("Display frame", sobelx);
	cv::waitKey(0);
	cv::imshow("Display frame", sobely);
	cv::waitKey(0);
	cv::imshow("Display frame", sobelxy);
	cv::waitKey(0);

	// Canny edge detection
	cv::Mat edges;
	cv::Canny(img_blur, edges, 100, 200, 3, false);
	
	// Display canny edge detected image
	cv::imshow("Display frame", edges);
	cv::waitKey(0);

	cv::destroyAllWindows();
	std::cout << "OpenCV Version: "<< CV_VERSION << std::endl;
*/
	// OpenMP + MPI
	int numprocs, rank, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int iam = 0, np = 1;
    omp_set_num_threads(numThreads);

    // Initialize MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);               // get number of processes
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);                   // get my process id
	MPI_Get_processor_name(processor_name, &namelen);       // get processor name

    printf ("Hello from process %d/%d on %s.\n", rank, numprocs, processor_name);

    if (numprocs >= 2) {

        int number;
        if (rank == 0)
        {
            number = -42;
            MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            printf("Process 0 sent number %d to process 1\n", number);
            #pragma omp parallel default(shared) private(iam, np)
            {
                np = omp_get_num_threads();
                iam = omp_get_thread_num();
                std::cout << "Hello from thread " << iam << " out of " << np <<
                          " from process " << rank << " out of " << numprocs << " on " <<
                          processor_name <<"\n";
            }
        }
        else if (rank == 1)
        {
            MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process 1 received number %d from process 0\n", number);
            #pragma omp parallel default(shared) private(iam, np)
            {
                np = omp_get_num_threads();
                iam = omp_get_thread_num();
                std::cout << "Hello from thread " << iam << " out of " << np <<
                          " from process " << rank << " out of " << numprocs << " on " <<
                          processor_name <<"\n";
            }
        }

    }
    /*
	#pragma omp parallel default(shared) private(iam, np)
	{
        np = omp_get_num_threads();
        iam = omp_get_thread_num();
        std::cout << "Hello from thread " << iam << " out of " << np <<
            " from process " << rank << " out of " << numprocs << " on " << processor_name <<"\n";
	}
*/
    // End MPI Environment
	MPI_Finalize();         // Terminates MPI environment

    // End of Program
	return 0;
}
