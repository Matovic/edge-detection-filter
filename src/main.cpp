/**
* Course: Parallel Programming (FIIT - WS 2021/2022)
* Purpose:
*
* @file main.cpp
* @author Erik Matovic
* @version 30.11.2021
*/

#include <iostream>
#include <string>
#include<sstream>
#include <tuple>
#include <mpi.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>

const uint8_t edge_width = 3;
const uint8_t  edge_height = 3;

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
std::tuple<int, int, int> processArguments(const int& argc, char* argv[])
{
    // equals -1 means they are all not valid
    int numThreads = -1, lowThreshold = -1, ratio = -1;
    std::vector<std::string> args(argv + 1, argc + argv);

    for (std::string consoleSwitch : args)
    {
        // make all characters uppercase in given string
        toUpper(consoleSwitch);

        // check given numbers on command line & set up if it is right
        if (numThreads == 0)
            numThreads = checkNumberOnConsoleSwitch(consoleSwitch);

        else if (lowThreshold == 0)
            lowThreshold = checkNumberOnConsoleSwitch(consoleSwitch);

        else if (ratio == 0)
            ratio = checkNumberOnConsoleSwitch(consoleSwitch);

        // check given command on command line is --THREADS & numThreads is not set up
        else if (consoleSwitch.compare("--THREADS") == 0 && numThreads == -1)
            numThreads = 0;

        // check given command on command line is --THRESHOLD & lowThreshold is not set up
        else if (consoleSwitch.compare("--THRESHOLD") == 0 && lowThreshold == -1)
            lowThreshold = 0;

        // check given command on command line is --RATIO & lowThreshold is not set up
        else if (consoleSwitch.compare("--RATIO") == 0 && ratio == -1)
            ratio = 0;

        // given command is wrong
        else
        {
            std::cerr << "ERROR 00: Wrong commands!\n";
            std::exit(EXIT_FAILURE);
        }
    }
    return std::make_tuple(numThreads, lowThreshold, ratio);
}

// Checks if given command line arguments are fit to be parameters for an Othello game.
int checkParameters(const int& numThreads, const int& lowThreshold, const int& ratio)
{
    if (numThreads < 1 && lowThreshold < 0 && ratio < 1)
    {
        std::cerr << "ERROR 01: Number of threads can not be less than 1!\n";
        return 1;
    }

    if (numThreads < 1 && (lowThreshold < 0 || ratio < 1))
    {
        std::cerr << "ERROR 01: Ratio can not be less than 1 & lowThreshold can not be less than 0!\n";
        return 1;
    }

    return 0;
}

// Get parameters from command line arguments.
std::tuple<int, int, int> getParameters(const int& argc, char* argv[])
{
    auto t = processArguments(argc, argv);
    //std::tie<numThreads, lowThreshold, ratio>
    int numThreads = std::get<0>(t), lowThreshold = std::get<1>(t), ratio = std::get<2>(t);
    //numThreads = processArguments(argc, argv);

    if (checkParameters(numThreads, lowThreshold, ratio))
        std::exit(EXIT_FAILURE);

    return std::make_tuple(numThreads, lowThreshold, ratio);
}

cv::Mat cannyEdgeDetectorOpenCV(const cv::Mat& img, const unsigned int& lowThreshold,
                                const unsigned int& ratio, const unsigned int& kernel_size)
{
    // Blur the image for better edge detection
    cv::Mat img_blur;
    cv::GaussianBlur(img, img_blur, cv::Size(3,3), 0);

    // Canny edge detection
    cv::Mat edges;
    // detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size
    cv::Canny(img_blur, edges, lowThreshold, lowThreshold * ratio, kernel_size, false);
    return edges;
}

// Computes the x component of the gradient vector
// at a given point in a image.
// returns gradient in the x direction
int xGradient(cv::Mat image, int x, int y)
{
    return image.at<int>(y-1, x-1) +
           2*image.at<int>(y, x-1) +
           image.at<int>(y+1, x-1) -
           image.at<int>(y-1, x+1) -
           2*image.at<int>(y, x+1) -
           image.at<int>(y+1, x+1);
}

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction

int yGradient(cv::Mat image, int x, int y)
{
    return image.at<int>(y-1, x-1) +
           2*image.at<int>(y-1, x) +
           image.at<int>(y-1, x+1) -
           image.at<int>(y+1, x-1) -
           2*image.at<int>(y+1, x) -
           image.at<int>(y+1, x+1);
}

cv::Mat cannyEdgeDetector(cv::Mat& img, const unsigned int& numThreads)
{
    // sum based on color of RGB color space
    float sumRed = 0, sumGreen = 0, sumBlue = 0;
    // convolution matrix
    float edge_filter[edge_width][edge_height] =
            {
                    //{-1, -1, -1},
                    //{-1, 8, -1},
                    //{-1, -1, -1}
                    {-0.125, -0.125, -0.125},
                    {-0.125, 1, -0.125},
                    {-0.125, -0.125, -0.125}
            };
    int edge_sum=0;   // Beware of the naughty zero
    int gx, gy, sum;

    cv::Mat dst = img.clone();

    auto img_width = img.size().width, img_height = img.size().height;

    for(int y = 0; y < img.rows; y++)
        for(int x = 0; x < img.cols; x++)
            dst.at<int>(y,x) = 0.0;

    for(int y = 1; y < img.rows - 1; y++){
        for(int x = 1; x < img.cols - 1; x++){
            gx = xGradient(img, x, y);
            gy = yGradient(img, x, y);
            sum = abs(gx) + abs(gy);
            sum = sum > 255 ? 255:sum;
            sum = sum < 0 ? 0 : sum;
            dst.at<int>(y,x) = sum;
        }
    }
/*
    for(unsigned int i = 1; i < img_width - 1; ++i)
    {
        for(unsigned int j = 1; j < img_height - 1; ++j)
        {
            for(unsigned int k = 0; k < edge_width; ++k)
            {
                for(unsigned int l = 0; l < edge_height; ++l)
                {
                    //auto color = getPixel(temp, i - ((edge_w-1) >> 1) + k, j - ((edge_h - 1) >> 1) + l);
                    // get pixel
                    //Vec3b color = image.at<Vec3b>(Point(x,y));
                    //cv::Vec3b pixelColor = img_gray.at<cv::Vec3b>(cv::Point(i,j));

                    cv::Vec3b pixelColor = img_gray.at<cv::Vec3b>(cv::Point(i-((edge_width-1)>>1)+k,j-((edge_height-1)>>1)+l));

                    uint8_t r = static_cast<uint8_t>(pixelColor[2]);
                    uint8_t g = static_cast<uint8_t>(pixelColor[1]);
                    uint8_t b = static_cast<uint8_t>(pixelColor[0]);

                    sumRed += r * edge_filter[k][l];
                    sumGreen += g * edge_filter[k][l];
                    sumBlue += b * edge_filter[k][l];
                }
            }

            // bring the color value y[r,c] back into 0-255
            //sumRed += 128;
            //sumGreen += 128;
            //sumBlue += 128;

            //sumRed *= 0.045;
            //sumGreen *= 0.045;
            //sumBlue *= 0.045;


            // set color
            cv::Vec3b color(sumBlue, sumGreen, sumRed);

            //putPixel(temp1, i, j, makeCol(sumRed, sumGreen, sumBlue));
            img_gray.at<cv::Vec3b>(cv::Point(i, j)) = color;
        }
    }*/
    return dst;
}

int main(int argc, char *argv[]) 
{
    // expected --THREADS <number of threads> or --THRESHOLD <lowThreashold> --RATIO <ratio>
	if (argc != 3 && argc != 5)
	{
		std::cerr << "ERROR 00: Invalid parameters\n";
		return 1;
	}

    //std::string filename = "data/lena.bmp";
    std::string filename = "data/4987_21_HE.tif";

    auto t = getParameters(argc, argv);
    //std::tie<numThreads, lowThreshold, ratio>
    int numThreads = std::get<0>(t), lowThreshold = std::get<1>(t), ratio = std::get<2>(t), kernel_size = edge_height;
    // std::cout << numThreads << '\n' << lowThreshold << '\n' << ratio << '\n';

    // Read image as colored image(1 as a flag)
    //cv::Mat img = cv::imread("data/4987_21_HE.tif", 1); // default Mat is CV_8UC3(8-bit 3-channel color image) matrix
    cv::Mat img = cv::imread(filename, 1);

    if (img.empty())
    {
        std::cerr << "ERROR 02: Could not open or find the image!\n";
        return EXIT_FAILURE;
    }

    // Convert to graycsale
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // Canny edge detection
    cv::Mat edges;
    // OpenCV Canny edge detection
    if (lowThreshold != -1 && ratio != 0)
    {
        edges = std::move(cannyEdgeDetectorOpenCV(img_gray, lowThreshold, ratio, kernel_size));
    }
    // Sobel edge detection
    else
        edges = std::move(cannyEdgeDetector(img_gray, numThreads));

    std::string str_grey = "Greyscale ";
    std::string str_edge= "Edge ";

	// Set window
    cv::namedWindow(str_edge, cv::WINDOW_NORMAL);
    cv::namedWindow(str_grey, cv::WINDOW_NORMAL);
    cv::namedWindow("Original image", cv::WINDOW_NORMAL);

	//Resize window
	cv::resizeWindow(str_edge, 512, 512);
    cv::resizeWindow(str_grey, 512, 512);
    cv::resizeWindow("Original image", 512, 512);

    // create new folder
    //std::string folderName = "output";
    //std::string folderCreateCommand = "mkdir " + folderName;
    //system(folderCreateCommand.c_str());

    // set filename output
    std::string filename_output = std::move(filename);
    filename_output.insert(filename_output.find("/") + 1, "output/");

    if (numThreads > 0)
    {
        std::string specification = "T" + std::to_string(numThreads);
        filename_output.insert(filename_output.find("."), specification);
    }
    else if (lowThreshold > 0 && ratio > 0)
    {
        std::string specification = "T" + std::to_string(lowThreshold) + "R" + std::to_string(ratio);
        filename_output.insert(filename_output.find("."), specification);
    }

    // full path
    std::stringstream ss;
    ss << filename_output; //<< folderName << "/" << filename_output;

    std::string fullPath = ss.str();
    ss.str("");

	// Display original image
    if (!img.empty() && !img_gray.empty() && !edges.empty())
    {
        cv::imshow("Original image", img);
        cv::imshow(str_grey, img_gray);
        cv::imshow(str_edge, edges);
        cv::waitKey(0);

        // Save the frame into a file
        //cv::imwrite(fullPath, edges);
        //std::cout << fullPath;
    }

    /*
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

    // End MPI Environment
	MPI_Finalize();         // Terminates MPI environment
    */
    cv::destroyAllWindows();
    // End of Program
	return 0;
}
