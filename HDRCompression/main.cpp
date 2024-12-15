#include <iostream>
#include "HDRCompression.h"
#include <opencv2/highgui.hpp>
int main()
{
    std::cout << "Hello world" << std::endl;


    cv::Mat test;
    cv::Mat image = cv::imread("C:\\Users\\giang\\Downloads\\bigFogMap.hdr", cv::IMREAD_UNCHANGED);
    cv::Mat log;
    cv::log(image, log);
    //cv::flip(image, image, 1);
    cv::Mat result;
    HDRCompression h(0.05, 0.8, 0.4);
    result = h.Apply(image, test);

    return 0;
}