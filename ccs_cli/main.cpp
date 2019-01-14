#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/timer.hpp>
#include "ccs_opencv.h"
#include "io_util.h"
#include "superpixel_tools.h"
#include "visualization.h"

/** \brief Command line tool for running CCS.
 * Usage:
 * \code{sh}
 *  $ ../bin/ccs_cli --help
 *  Allowed options:
 *    -h [ --help ]                   produce help message
 *    -i [ --input ] arg              the folder to process
 *    -s [ --superpixels ] arg (=400) number of superpixels
 *    -c [ --compactness ] arg (=500) compactness weight
 *    -t [ --iterations ] arg (=20)   number of iterations to perform
 *    -r [ --color-space ] arg (=0)   0 = RGB, >0 = Lab
 *    -o [ --csv ] arg                save segmentation as CSV file
 *    -v [ --vis ] arg                visualize contours
 *    -x [ --prefix ] arg             output file prefix
 *    -w [ --wordy ]                  verbose/wordy/debug
 * \endcode
 * \author David Stutz
 */
int main(int argc, const char** argv) {
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input,i", boost::program_options::value<std::string>(), "input picture")
        ("superpixels,s", boost::program_options::value<int>()->default_value(400), "number of superpixels")
        ("compactness,c", boost::program_options::value<int>()->default_value(500), "compactness weight")
        ("iterations,t", boost::program_options::value<int>()->default_value(20), "number of iterations to perform")
        ("color-space,r", boost::program_options::value<int>()->default_value(0), "0 = RGB, >0 = Lab")
        ("oc", boost::program_options::value<std::string>()->default_value("output"), "name of the contour picture")
        ("om", boost::program_options::value<std::string>()->default_value("output"), "name of the mean picture");

    boost::program_options::positional_options_description positionals;
    positionals.add("input", 1);
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end()) {
        std::cout << desc << std::endl;
        return 1;
    }

    std::string inputfile = parameters["input"].as<std::string>();
    std::string store_contour = parameters["oc"].as<std::string>();
    std::string store_mean = parameters["om"].as<std::string>();
    int superpixels = parameters["superpixels"].as<int>();
    int compactness = parameters["compactness"].as<int>();
    int iterations = parameters["iterations"].as<int>();
    int color_space_int = parameters["color-space"].as<int>();
    
    bool lab = false;
    if (color_space_int > 0) {
        lab = true;
    }
        
    cv::Mat image = cv::imread(inputfile);
    cv::Mat labels;
        
    int region_size = SuperpixelTools::computeRegionSizeFromSuperpixels(image, 
                superpixels);
        
    CCS_OpenCV::computeSuperpixels(image, region_size,
                iterations, compactness, lab, labels);
        
    int unconnected_components = SuperpixelTools::relabelConnectedSuperpixels(labels);
    int merged_components = SuperpixelTools::enforceMinimumSuperpixelSizeUpTo(image, labels, unconnected_components);
    merged_components += SuperpixelTools::enforceMinimumSuperpixelSizeUpTo(image, labels, unconnected_components);
      
    cv::Mat blackima = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);
    
    cv::Mat image_contours;
    Visualization::drawContours(blackima, labels, image_contours);
    cv::imwrite(store_contour, image_contours);

    cv::Mat image_means;
    Visualization::drawMeans(image, labels, image_means);
    cv::imwrite(store_mean, image_means);
            
    return 0;
}
