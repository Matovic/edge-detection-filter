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
            //MPI_Finalize();         // Terminates MPI environment
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
    {
        //MPI_Finalize();         // Terminates MPI environment
        std::exit(EXIT_FAILURE);
    }

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

int main(int argc, char *argv[])
{
    // expected --THREADS <number of threads> --THRESHOLD <lowThreashold> --RATIO <ratio>
    if (argc != 7)
    {
        std::cerr << "ERROR 00: Invalid parameters\n";
        return 1;
    }

    int numProc, rank = 0, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int thread_id = 0, num_threads = 1;                     // num_threads is total threads

    MPI_Init(&argc, &argv);                                 // initialize MPI
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);                // get number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                   // get my process id
    MPI_Get_processor_name(processor_name, &namelen);       // get processor name

    // get parameters
    auto t = getParameters(argc, argv);
    int numThreads = std::get<0>(t), lowThreshold = std::get<1>(t), ratio = std::get<2>(t), kernel_size = edge_height;
    omp_set_num_threads(numThreads);    // set number of threads if it was given

    // vectors of images
    std::vector<cv::Mat> v_img;
    std::vector<cv::Mat> v_out_img;
    std::vector<std::string> v_fullpath;

    std::string path = "data/";                 // path of input folder
    std::string out_dir = "./output";           // path for output directory

    std::vector<std::string> files;             // vector of paths of input files
    std::vector<std::string> output_folders;    // vector of output folders where to store images

    // get all files
    auto itr = std::filesystem::directory_iterator(path);   // iterator
    int counter = 0;
    for (auto entry = begin(itr); begin(entry) != end(itr); ++entry, ++counter)
    {
        std::string file = entry->path();           // get path of a directory or of a file
        if (std::filesystem::is_directory(file))    // check if string is not a directory
            continue;                               // skip directories

        // files are processed by different processors because of an MPI
        if (counter % numProc != rank)
            continue;

        // file is already in the vector
        if (std::find(files.begin(), files.end(), file) != files.end())
            continue;
        files.push_back(file);                      // push back file into the vector
    }

    // if output directory does not exist, create one
    if (!std::filesystem::is_directory(out_dir))
    {
        // create directory to store files into
        std::string folderCreateCommand = "mkdir " + out_dir;
        system(folderCreateCommand.c_str());
    }

    // extract name of input files in order to create output folders
    for (int i = 0; i < files.size(); ++i)
    {
        // extract name of a file
        size_t pos = files[i].find("/") + 1;    // in order to get rid of folder
        size_t end_pos = files[i].find(".");    // in order to get rid of suffix
        size_t len = end_pos - pos;                // in order to get file's name without suffix

        // create new folder name based on name of a file
        std::string folderName = out_dir + '/' + files[i].substr(pos,len);

        // if file is already in the vector
        if (std::find(output_folders.begin(), output_folders.end(), folderName) != output_folders.end())
            continue;
        else output_folders.push_back(folderName);  // create directory to store files into
    }

    // create output folders based on name of files
    #pragma omp parallel default(shared)
    {
        int i;
        #pragma omp for private(i)
        for (i = 0; i < output_folders.size(); ++i) {
            // if output directory does not exist, create one
            if (!std::filesystem::is_directory(output_folders[i])) {
                std::string folderCreateCommand = "mkdir " + output_folders[i];
                system(folderCreateCommand.c_str());
            }
        }
    }

    // map input image with its output folder
    for (int i = 0; i < files.size(); i++)
    {
        // extract name of a file
        size_t pos = files[i].find("/") + 1;        // in order to get rid of folder
        size_t end_pos = files[i].find(".");        // in order to get rid of suffix
        size_t len = end_pos - pos;                    // in order to get name from file

        // find output file
        std::string o_folder = out_dir + '/' + files[i].substr(pos,len);
        std::vector<std::string>::iterator it = std::find(output_folders.begin(), output_folders.end(), o_folder);

        // find output folder's index in output_folders vector
        int indexOut = 0;
        if (it != output_folders.end())
            indexOut = std::distance(output_folders.begin(), it);
        else
        {
            std::cerr << "ERROR 04: Could not map input image with its output folder!\n";
            return EXIT_FAILURE;
        }
        //std::cout << files[i] << " out: "<< output_folders[indexOut] << '\n';

        // get output's file name
        len = files[i].size() - pos;
        std::string filename_output = files[i].substr(pos,len);
        std::string specification = "_LT" + std::to_string(lowThreshold) + "_R" + std::to_string(ratio);
        filename_output.insert(filename_output.find("."), specification);

        // set full path of output with new name and output folder
        std::stringstream ss;
        ss << output_folders[indexOut] << "/" << filename_output;

        std::string fullPath = ss.str();
        ss.str("");
        //std::cout << "Full Path: " << fullPath << '\n';

        // file is already in the vector
        if (std::find(v_fullpath.begin(), v_fullpath.end(), fullPath) != v_fullpath.end())
            continue;
        v_fullpath.push_back(fullPath);
    }

    #pragma omp parallel default(shared) private(thread_id)
    {
        num_threads = omp_get_num_threads();            // get all threads
        thread_id = omp_get_thread_num();               // get thread id
        int i;
        #pragma omp for private(i)
        for (i = 0; i < files.size(); ++i)
        {
            // Read image as colored image(1 as a flag)
            cv::Mat img = cv::imread(files[i], 1);

            // Convert to graycsale
            cv::Mat img_gray;
            cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

            // Canny edge detection
            cv::Mat edges = std::move(cannyEdgeDetector(img_gray, lowThreshold, ratio, kernel_size));

            // Save img
            cv::imwrite(v_fullpath[i], edges);
            std::cout << "Save path: " << v_fullpath[i] << " Input: " << files[i] <<
                      " rank " << rank << "/" << numProc << " thread " << thread_id << "/" << num_threads << '\n';
        }
        //std::cout << "rank " << rank << "/" << numProc << " thread " << thread_id << "/" << num_threads << '\n';
    }
    MPI_Finalize();         // Terminates MPI environment
    return 0;
}
