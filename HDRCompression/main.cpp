#include <iostream>
#include "HDRCompression.h"
#include <opencv2/highgui.hpp>
int main()
{
    std::cout << "Hello world" << std::endl;


    cv::Mat test;
    cv::Mat image = cv::imread(".\\images\\cathedral.hdr", cv::IMREAD_UNCHANGED);
    cv::Mat log;
    cv::log(image, log);
    //cv::flip(image, image, 1);
    cv::Mat result;
    HDRCompression h(0.1, 0.8, 0.6);
    result = h.Apply(image, test);
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, CV_32FC1);
    cv::exp(result, result);
    return 0;
}