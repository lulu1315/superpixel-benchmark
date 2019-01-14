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
#include "graph_segmentation.h"
#include "io_util.h"
#include "visualization.h"
#include "superpixel_tools.h"

/** \brief Command line tool for running reFH.
 * Usage:
 * \code{sh}
 *   $ ../bin/refh_cli --help
 *   Allowed options:
 *     -h [ --help ]                   produce help message
 *     -i [ --input ] arg              folder containing the images to process
 *     -g [ --sigma ] arg (=0)         sigma used for smoothing (no smoothing if 
 *                                     zero)
 *     -t [ --threshold ] arg (=20)    constant for threshold function
 *     -m [ --minimum-size ] arg (=10) minimum component size
 *     -o [ --csv ] arg                save segmentation as CSV file
 *     -v [ --vis ] arg                visualize contours
 *     -x [ --prefix ] arg             output file prefix
 *     -w [ --wordy ]                  verbose/wordy/debug
 * \endcode
 * \author David Stutz
 */
int main (int argc, char ** argv) {
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input,i", boost::program_options::value<std::string>(), "folder containing the images to process")
        ("sigma,g", boost::program_options::value<float>()->default_value(0.0f), "sigma used for smoothing (no smoothing if zero)")
        ("threshold,t", boost::program_options::value<float>()->default_value(20.0f), "constant for threshold function")
        ("minimum-size,m", boost::program_options::value<int>()->default_value(10), "minimum component size")
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
    float sigma = parameters["sigma"].as<float>();
    float threshold = parameters["threshold"].as<float>();
    int minimum_segment_size = parameters["minimum-size"].as<int>();
            
    cv::Mat image = cv::imread(inputfile);
        
    // See lib_fh/filter.h
    if (sigma > 0.01) {
        int size = std::ceil(sigma*4) + 1;
        cv::GaussianBlur(image, image, cv::Size (size, size), sigma, sigma);
    }
        
    GraphSegmentationMagicThreshold magic(threshold);
    GraphSegmentationEuclideanRGB distance;
        
    GraphSegmentation segmenter;
    segmenter.setMagic(&magic);
    segmenter.setDistance(&distance);
        
    segmenter.buildGraph(image);
    segmenter.oversegmentGraph();
        
    segmenter.enforceMinimumSegmentSize(minimum_segment_size);
        
    cv::Mat labels = segmenter.deriveLabels();
        
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
