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
std::tuple<int, int, int, int> processArguments(const int& argc, char* argv[])
{
    // numThreads = -1, maxDepth = -1, heuristic = -1, moveTime = -1 are all not valid
    int numThreads = -1, maxDepth = -1, heuristic = -1, moveTime = -1;
    std::vector<std::string> args(argv + 1, argc + argv);

    for (std::string consoleSwitch : args)
    {
        // make all characters uppercase in given string.
        toUpper(consoleSwitch);
        std::cout << consoleSwitch << '\n';
        // check given numbers on command line
        if (numThreads == 0)
            numThreads = checkNumberOnConsoleSwitch(consoleSwitch);

        /*else if (maxDepth == 0)
            maxDepth = checkNumberOnConsoleSwitch(consoleSwitch);

        else if (heuristic == 0)
            heuristic = checkNumberOnConsoleSwitch(consoleSwitch);

        else if (moveTime == 0)
            moveTime = checkNumberOnConsoleSwitch(consoleSwitch);*/

        // check given commands on command line
        else if (consoleSwitch.compare("--THREADS") == 0 && numThreads == -1)
            numThreads = 0;
        /*
        else if (consoleSwitch.compare("--TIME") == 0 && moveTime == -1)
            moveTime = 0;

        else if (consoleSwitch.compare("--FUNC") == 0 && heuristic == -1)
            heuristic = 0;

        else if (consoleSwitch.compare("--DEPTH") == 0 && maxDepth == -1)
            maxDepth = 0;*/

        else
        {
            std::cerr << "ERROR 00: Wrong parameters!\n";
            std::exit(EXIT_FAILURE);
        }
    }
    return std::make_tuple(numThreads, maxDepth, heuristic, moveTime);
}

// Checks if given command line arguments are fit to be parameters for an Othello game.
int checkParameters(const int& numThreads, const int& maxDepth, const int& heuristic, const int& moveTime)
{
    if (numThreads < 1)
    {
        std::cerr << "ERROR 01: Number of threads can not be less than 1!\n";
        return 1;
    }
/*
    if (maxDepth < 1 || maxDepth > 15)
    {
        std::cerr << "ERROR 02: Incorrect depth! Depth can be from 5 to 15.\n";
        return 1;
    }

    if (heuristic < 1 || heuristic > 2)
    {
        std::cerr << "ERROR 03: Heuristic function is not known!\n";
        return 1;
    }

    if (moveTime < 5 || moveTime > 30)
    {
        std::cerr << "ERROR 04: Incorrect time! Time can be from 5 to 30.\n";
        return 1;
    }
    */
    return 0;
}

// Get parameters from command line arguments.
std::tuple<int, int, int, int> getParameters(const int& argc, char* argv[])
{
    // numThreads = -1, maxDepth = -1, heuristic = -1, moveTime = -1 are all not valid
    int numThreads = -1, maxDepth = -1, heuristic = -1, moveTime = -1;
    std::tie(numThreads, maxDepth, heuristic, moveTime) = processArguments(argc, argv);

    if (maxDepth == -1) maxDepth = 15;
    if (heuristic == -1) heuristic = 2;
    if (moveTime == -1) moveTime = 15;

    if (checkParameters(numThreads, maxDepth, heuristic, moveTime))
        std::exit(EXIT_FAILURE);

    return std::make_tuple(numThreads, maxDepth, heuristic, moveTime);
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
    int numThreads = -1, maxDepth = -1, heuristic = -1, moveTime = -1;
    std::tie(numThreads, maxDepth, heuristic, moveTime) = getParameters(argc, argv);

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

	#pragma omp parallel default(shared) private(iam, np)
	{
        np = omp_get_num_threads();
        iam = omp_get_thread_num();
        std::cout << "Hello from thread " << iam << " out of " << np <<
            " from process " << rank << " out of " << numprocs << " on " << processor_name <<"\n";
	}

    // End MPI Environment
	MPI_Finalize();         // Terminates MPI environment

    // End of Program
	return 0;
}
