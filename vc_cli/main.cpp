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
#include "vc_opencv.h"
#include "io_util.h"
#include "superpixel_tools.h"
#include "visualization.h"

/** \brief Command line tool for running VC.
 * Usage:
 * \code{sh}
 *   $ ../bin/vc_cli --help
 *   Allowed options:
 *     -h [ --help ]                         produce help message
 *     -i [ --input ] arg                    folder containing the images to process
 *     -s [ --superpixels ] arg (=400)       number superpixels
 *     -c [ --weight ] arg (=5)              number superpixels
 *     -g [ --radius ] arg (=3)              radius
 *     -n [ --neighboring-clusters ] arg (=200)
 *                                           number of neighboring clusters
 *     -d [ --direct-neighbors ] arg (=4)    number of direct neighbors
 *     -t [ --threshold ] arg (=10)          threshold influencing the number of 
 *                                           iterations
 *     -r [ --color-space ] arg (=1)         color space; 0 for RGB, > 0 for Lab
 *     -o [ --csv ] arg                      save segmentation as CSV file
 *     -v [ --vis ] arg                      visualize contours
 *     -x [ --prefix ] arg                   output file prefix
 *     -w [ --wordy ]                        verbose/wordy/debug
 * \endcode
 * \author David Stutz
 */
int main (int argc, char ** argv) {
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input,i", boost::program_options::value<std::string>(), "folder containing the images to process")
        ("superpixels,s", boost::program_options::value<int>()->default_value(400), "number superpixels")
        ("weight,c", boost::program_options::value<double>()->default_value(5.0), "number superpixels")
        ("radius,g", boost::program_options::value<int>()->default_value(3), "radius")
        ("neighboring-clusters,n", boost::program_options::value<int>()->default_value(200), "number of neighboring clusters")
        ("direct-neighbors,d", boost::program_options::value<int>()->default_value(4), "number of direct neighbors")
        ("threshold,t", boost::program_options::value<int>()->default_value(10), "threshold influencing the number of iterations")
        ("color-space,r", boost::program_options::value<int>()->default_value(1), "color space; 0 for RGB, > 0 for Lab")
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
    double weight = parameters["weight"].as<double>();
    int radius = parameters["radius"].as<int>();
    int neighboring_clusters = parameters["neighboring-clusters"].as<int>();
    int direct_neighbors = parameters["direct-neighbors"].as<int>();
    int threshold = parameters["threshold"].as<int>();
           
    cv::Mat image = cv::imread(inputfile);
    cv::Mat labels;
        
    VC_OpenCV::computeSuperpixels(image, superpixels, weight, radius, 
                neighboring_clusters, direct_neighbors, threshold, labels);
        
    int unconnected_components = SuperpixelTools::relabelConnectedSuperpixels(labels);
    int merged_unconnected_components = SuperpixelTools::enforceMinimumSuperpixelSizeUpTo(image, labels, unconnected_components);
        
    int size = (float) (image.rows*image.cols)/superpixels/10;
    int merged_small_components = SuperpixelTools::enforceMinimumSuperpixelSize(image, labels, size);
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
