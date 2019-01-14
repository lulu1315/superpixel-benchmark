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
#include <bitset>
#include "slic_opencv.h"
#include "io_util.h"
#include "superpixel_tools.h"
#include "visualization.h"

/** \brief Command line tool for running SEEDS.
 * Usage:
 * \code{sh}
 *   $ ../bin/slic_cli --help
 *   Allowed options:
 *     -h [ --help ]                   produce help message
 *     -i [ --input ] arg              the folder to process (can also be passed as 
 *                                     positional argument)
 *     -s [ --superpixels ] arg (=400) number of superpixles
 *     -c [ --compactness ] arg (=40)  compactness
 *     -p [ --perturb-seeds ] arg (=1) perturb seeds: > 0 yes, = 0 no
 *     -t [ --iterations ] arg (=10)   iterations
 *     -r [ --color-space ] arg (=1)   color space: 0 = RGB, > 0 = Lab
 *     -o [ --csv ] arg                specify the output directory (default is 
 *                                     ./output)
 *     -v [ --vis ] arg                visualize contours
 *     -x [ --prefix ] arg             output file prefix
 *     -w [ --wordy ]                  verbose/wordy/debug
 * \endcode
 * \author David Stutz
 */
int main(int argc, const char** argv) {
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input,i", boost::program_options::value<std::string>(), "the folder to process (can also be passed as positional argument)")
        ("superpixels,s", boost::program_options::value<int>()->default_value(400), "number of superpixles")
        ("compactness,c", boost::program_options::value<double>()->default_value(40.), "compactness")
        ("perturb-seeds,p", boost::program_options::value<int>()->default_value(1), "perturb seeds: > 0 yes, = 0 no")
        ("iterations,t", boost::program_options::value<int>()->default_value(10), "iterations")
        ("color-space,r", boost::program_options::value<int>()->default_value(1), "color space: 0 = RGB, > 0 = Lab")
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
    double compactness = parameters["compactness"].as<double>();
    int iterations = parameters["iterations"].as<int>();
    int perturb_seeds_int = parameters["perturb-seeds"].as<int>();
    bool perturb_seeds = perturb_seeds_int > 0 ? true : false;
    int color_space = parameters["color-space"].as<int> ();
            
    cv::Mat image = cv::imread(inputfile);
    cv::Mat labels;
        
    int region_size = SuperpixelTools::computeRegionSizeFromSuperpixels(image, 
            superpixels);
        
    SLIC_OpenCV::computeSuperpixels(image, region_size, compactness, 
            iterations, perturb_seeds, color_space, labels);
        
    int unconnected_components = SuperpixelTools::relabelConnectedSuperpixels(labels);
        
    cv::Mat blackima = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);
    
    cv::Mat image_contours;
    Visualization::drawContours(blackima, labels, image_contours);
    cv::imwrite(store_contour, image_contours);

    cv::Mat image_means;
    Visualization::drawMeans(image, labels, image_means);
    cv::imwrite(store_mean, image_means);  
    
    return 0;
}
