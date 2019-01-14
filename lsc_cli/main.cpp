/**
 * Copyright (c) 2016, David Stutz
 * Contact: david.stutz@rwth-aachen.de, davidstutz.de
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/timer.hpp>
#include "lsc_opencv.h"
#include "io_util.h"
#include "superpixel_tools.h"
#include "visualization.h"

/** \brief Command line tool for running LSC.
 * Usage:
 * \code{sh}
 *   $ ../bin/lsc_cli --help
 *   Allowed options:
 *     -h [ --help ]                         produce help message
 *     -i [ --input ] arg                    the folder to process
 *     -s [ --superpixels ] arg (=400)       number of superpixels
 *     -c [ --ratio ] arg (=0.074999999999999997)
 *                                           compactness ratio = color weight / 
 *                                           spatial weight
 *     -t [ --iterations ] arg (=20)         number of iterations to perform
 *     -g [ --threshold ] arg (=4)           threshold coefficient
 *     -r [ --color-space ] arg (=1)         color space: 0 = RGB, >0 = Lab
 *     -f [ --fair ]                         for a fair comparison with other 
 *                                           algorithms, quadratic blocks are used 
 *                                           for initialization
 *     -o [ --csv ] arg                      save segmentation as CSV file
 *     -v [ --vis ] arg                      visualize contours
 *     -x [ --prefix ] arg                   output file prefix
 *     -w [ --wordy ]                        verbose/wordy/debug
 * \endcode
 * \author David Stutz
 */
int main(int argc, const char** argv) {
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input,i", boost::program_options::value<std::string>(), "the folder to process")
        ("superpixels,s", boost::program_options::value<int>()->default_value(400), "number of superpixels")
        ("ratio,c", boost::program_options::value<double>()->default_value(0.075), "compactness ratio = color weight / spatial weight")
        ("iterations,t", boost::program_options::value<int>()->default_value(20), "number of iterations to perform")
        ("threshold,g", boost::program_options::value<int>()->default_value(4), "threshold coefficient")
        ("color-space,r", boost::program_options::value<int>()->default_value(1), "color space: 0 = RGB, >0 = Lab")
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
    double ratio = parameters["ratio"].as<double>();
    int iterations = parameters["iterations"].as<int>();
    int threshold = parameters["threshold"].as<int>();
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
        
    LSC_OpenCV::computeSuperpixels(image, region_height, region_width, ratio, 
            iterations, threshold, color_space, labels);
        
    int unconnected_components = SuperpixelTools::relabelConnectedSuperpixels(labels);
//  int merged_components = SuperpixelTools::enforceMinimumSuperpixelSize(image, labels, 5);
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
