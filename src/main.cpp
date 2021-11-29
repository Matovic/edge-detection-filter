/**
* Course: Parallel Programming (FIIT - WS 2021/2022)
* Purpose:
*
* @file main.cpp
* @author Erik Matovic
* @version 30.11.2021
*/

#include <iostream>
#include <vector>
#include <filesystem>
#include <string>
#include <sstream>
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
    if (numThreads < 1)
    {
        std::cerr << "ERROR 01: Number of threads can not be less than 1!\n";
        return 1;
    }

    if (lowThreshold < 1)
    {
        std::cerr << "ERROR 01: LowThreshold can not be less than 0!\n";
        return 1;
    }

    if (ratio < 1)
    {
        std::cerr << "ERROR 01: Ratio can not be less than 0!\n";
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

cv::Mat cannyEdgeDetector(const cv::Mat& img, const unsigned int& lowThreshold,
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

// https://programming-techniques.com/2013/03/sobel-and-prewitt-edge-detector-in-c-image-processing.html
cv::Mat sobelEdgeDetector(cv::Mat& img, const unsigned int& numThreads)
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
    return dst;
}

int main(int argc, char *argv[]) 
{
    // expected --THREADS <number of threads> or --THRESHOLD <lowThreashold> --RATIO <ratio>
    // expected --THREADS <number of threads> --THRESHOLD <lowThreashold> --RATIO <ratio>
    if (argc != 7) //if (argc != 3 && argc != 5)
	{
		std::cerr << "ERROR 00: Invalid parameters\n";
		return 1;
	}

    int numProc, rank = 0, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int iam = 0, np = 1;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);               // get number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                  // get my process id
    MPI_Get_processor_name(processor_name, &namelen);      // get processor name

    // get parameters
    auto t = getParameters(argc, argv);
    int numThreads = std::get<0>(t), lowThreshold = std::get<1>(t), ratio = std::get<2>(t), kernel_size = edge_height;

    // set number of threads if it was given
    omp_set_num_threads(numThreads);

    std::string path = "data/";         // path of input folder
    std::vector<std::string> files;     // vector of input files
    auto itr = std::filesystem::directory_iterator(path);

    // get all files
    #pragma omp parallel default(shared)
    {
        auto entry = begin(itr);
        #pragma omp critical            // writing to a vector, more threads can read same file, etc.
        {
            for (; begin(entry) != end(itr); ++entry)
            {
                //if (omp_get_thread_num() == 0)
                //    std::cout << "Threads:" << omp_get_num_threads() << std::endl;

                std::string file = entry->path();           // get path of a directory or of a file
                if (std::filesystem::is_directory(file))    // check if string is not a directory
                    continue;                               // skip directories
                //std::cout << "name: " << file << '\n';
                //std::cout << "Thread num: " << omp_get_thread_num() << '\n';

                // file is already in the vector
                if (std::find(files.begin(), files.end(), file) != files.end())
                    continue;
                files.push_back(file);                      // push back file into the vector
            }
        }
    }

    /*std::cout << files.size();
    for (int i = 0; i < files.size() - 1; ++i)
        std::cout << files[i] << '\n';*/

    // output directory
    std::string out_dir = "./output";

    // directory does not exist
    if (!std::filesystem::is_directory(out_dir))
    {
        // create directory to store files into
        std::string folderCreateCommand = "mkdir " + out_dir;
        //std::cout << folderCreateCommand << '\n';
        system(folderCreateCommand.c_str());
    }

    std::vector<std::string> output_folders;
    #pragma omp parallel default(shared)
    {
        int i = 0;
        #pragma omp for private(i)
        for (i = 0; i < files.size() - 1; ++i)
        {
            // extract name of a file
            size_t pos = files[i].find("/") + 1;
            size_t end_pos = files[i].find(".");
            size_t len = end_pos - pos;

            // create new folder based on name of a file
            std::string folderName = out_dir + '/' + files[i].substr(pos,len);
            if (std::find(files.begin(), files.end(), folderName) != files.end())
                continue;           // file is already in the vector
            else                    // create directory to store files into
            #pragma omp critical
            {
                output_folders.push_back(folderName);
            }
        }
    }

    /*for(const auto& f : output_folders)
        std::cout << f << '\n';*/

    #pragma omp parallel default(shared)
    {
        int i = 0;
        #pragma omp for private(i)
        for (i = 0; i < output_folders.size(); ++i)
        {
            // directory does not exist
            if (!std::filesystem::is_directory(output_folders[i])) {
                std::string folderCreateCommand = "mkdir " + output_folders[i];
                system(folderCreateCommand.c_str());
            }
        }
    }

    // process files
    for (int i = 0; i < files.size() - 1; i++)
    {
        // files are processed by different processors because of an MPI
        if (i % numProc != rank)
            continue;
        //printf ("Hello from process %d/%d on %s.\n", rank, numProc, processor_name);

        // extract name of a file
        size_t pos = files[i].find("/") + 1;
        size_t end_pos = files[i].find(".");
        size_t len = end_pos - pos;

        // find output file
        std::string o_folder = out_dir + '/' + files[i].substr(pos,len);
        std::vector<std::string>::iterator it = std::find(output_folders.begin(), output_folders.end(),
                                                          o_folder);
        //std::cout << "o_folder: " << o_folder << '\n';
        int indexOut = 0;
        if (it != output_folders.end())
            indexOut = std::distance(output_folders.begin(), it);
        else
        {
            std::cerr << "ERROR 04: Could not map input image with its output folder!\n";
            return EXIT_FAILURE;
        }
        //std::cout << files[i] << " out: "<< output_folders[indexOut] << '\n';

        // get new file name
        len = files[i].size() - pos;
        std::string filename_output = files[i].substr(pos,len);//files[i];
        std::string specification = "_P" + std::to_string(rank) + "_T" + std::to_string(numThreads) +
                "_LT" + std::to_string(lowThreshold) + "_R" + std::to_string(ratio);
        filename_output.insert(filename_output.find("."), specification);

        // full path
        std::stringstream ss;
        ss << output_folders[indexOut] << "/" << filename_output;

        std::string fullPath = ss.str();
        ss.str("");
        //std::cout << "Full Path: " << fullPath << '\n';

        // Read image as colored image(1 as a flag)
        //cv::Mat img = cv::imread("data/4987_21_HE.tif", 1); // default Mat is CV_8UC3(8-bit 3-channel color image) matrix
        cv::Mat img = cv::imread(files[i], 1);

        if (img.empty())
        {
            std::cerr << "ERROR 02: Could not open or find the image!\n";
            return EXIT_FAILURE;
        }

        // Convert to graycsale
        cv::Mat img_gray;
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

        // Canny edge detection
        cv::Mat edges = std::move(cannyEdgeDetector(img_gray, lowThreshold, ratio, kernel_size));
/*
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
*/
/*
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

        // Display original image
        if (!img.empty() && !img_gray.empty() && !edges.empty())
        {
*/
            /*cv::imshow("Original image", img);
            cv::imshow(str_grey, img_gray);
            cv::imshow(str_edge, edges);
            cv::waitKey(0);*/
/*
            // Save the frame into a file
            cv::imwrite(fullPath, edges);
            //std::cout << fullPath;
        }
*/
        // Save the frame into a file
        cv::imwrite(fullPath, edges);
    }
	MPI_Finalize();         // Terminates MPI environment
    //cv::destroyAllWindows();
	return 0;
}
