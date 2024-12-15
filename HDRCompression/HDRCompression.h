#pragma once
#include <opencv2/core.hpp>
class HDRCompression
{
public:
	HDRCompression(const float& alpha, const float& beta, const float& staturation = 0.6) : alpha(alpha), beta(beta), staturation(staturation) {};
	
	// Apply HDR compression
	// @return 0 if success other if error
	//
	cv::Mat Apply(const cv::Mat &image, cv::Mat &output);

private:
	cv::Mat CalculateGradient(const cv::Mat& image);
	cv::Mat CalculateScaleRate(const cv::Mat& gradientMagnitude, float trueAlpha);
	cv::Mat Color(const cv::Mat& originImage, const cv::Mat& ychannel, const cv::Mat& newIntensity, cv::Mat& output);
private: 
	float alpha;
	float beta;
	float staturation;

	int pyramidLevelNeeded = 3;
};

