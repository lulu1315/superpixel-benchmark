#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/timer.hpp>
#include "crs_opencv.h"
#include "io_util.h"
#include "superpixel_tools.h"
#include "visualization.h"

/** \brief Command line tool for running CRS.
 * Usage:
 * \code{sh}
 *   $ ../bin/crs_cli --help
 *   Allowed options:
 *     -h [ --help ]                         produce help message
 *     -i [ --input ] arg                    the folder to process
 *     -s [ --superpixels ] arg (=400)       number of superpixels
 *     -c [ --compactness ] arg (=0.044999999999999998)
 *                                           compactness weight
 *     -l [ --clique-cost ] arg (=0.29999999999999999)
 *                                           direct clique cost
 *     -t [ --iterations ] arg (=3)          number of iterations to perform
 *     -r [ --color-space ] arg (=0)         color space: 0 = YCrCb, 1 = RGB
 *     -f [ --fair ]                         for a fair comparison with other 
 *                                           algorithms, quadratic blocks are used 
 *                                           for initialization
 *     -o [ --csv ] arg                      save segmentation as CSV file
 *     -v [ --vis ] arg                      visualize contours
 *     -x [ --prefix ] arg                   output file prefix
 *     -w [ --wordy ]                        verbose/wordy/debug
 * \encode
 * \author David Stutz
 */
int main(int argc, const char** argv) {
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input,i", boost::program_options::value<std::string>(), "the folder to process")
        ("superpixels,s", boost::program_options::value<int>()->default_value(400), "number of superpixels")
        ("compactness,c", boost::program_options::value<double>()->default_value(0.045), "compactness weight")
        ("clique-cost,l", boost::program_options::value<double>()->default_value(0.3),  "direct clique cost")
        ("iterations,t", boost::program_options::value<int>()->default_value(3), "number of iterations to perform")
        ("color-space,r", boost::program_options::value<int>()->default_value(0), "color space: 0 = YCrCb, 1 = RGB")
        ("fair,f", "for a fair comparison with other algorithms, quadratic blocks are used for initialization")
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
    double clique_cost = parameters["clique-cost"].as<double>();
    double compactness = parameters["compactness"].as<double>();
    int iterations = parameters["iterations"].as<int>();
    int color_space = parameters["color-space"].as<int>();
    
    if (color_space < 0 || color_space > 1) {
        std::cout << "Invalid color space." << std::endl;
        return 1;
    }
        
    cv::Mat image = cv::imread(inputfile);
    cv::Mat labels;
        
    int region_width;
    int region_height;
    SuperpixelTools::computeHeightWidthFromSuperpixels(image, superpixels,
            region_height, region_width);
        
    // If a fair comparison is requested:
    if (parameters.find("fair") != parameters.end()) {
        region_width = SuperpixelTools::computeRegionSizeFromSuperpixels(image, 
                superpixels);
        region_height = region_width;
    }

    CRS_OpenCV::computeSuperpixels(image, region_height, region_width, clique_cost, 
            compactness, iterations, color_space, labels);
        
    int unconnected_components = SuperpixelTools::relabelConnectedSuperpixels(labels);
//    int merged_components = SuperpixelTools::enforceMinimumSuperpixelSize(image, labels, 5);
    int merged_components = SuperpixelTools::enforceMinimumSuperpixelSizeUpTo(image, labels, unconnected_components);
    SuperpixelTools::relabelSuperpixels(labels);
    
    cv::Mat blackima = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);
    
    cv::Mat image_contours;
    Visualization::drawContours(blackima, labels, image_contours);
    cv::imwrite(store_contour, image_contours);

    cv::Mat image_means;
    Visualization::drawMeans(image, labels, image_means);
    cv::imwrite(store_mean, image_means);
    
    return 0;
}
