#include "HDRCompression.h"
#include <mkl.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
cv::Mat PoissonSolver(const cv::Mat& laplacian) {
	cv::Mat result = laplacian.clone() * -1;

	float* laplacianResult = (float*)result.datastart;
	float ax = 0;
	float ay = 0;
	float bx = result.cols - 1;
	float by = result.rows - 1;
	int nx = laplacian.cols - 1;
	int ny = laplacian.rows - 1;
	char BCType[4] = { 'N', 'N', 'N', 'N' };
	float q = 0;
	DFTI_DESCRIPTOR_HANDLE xhandle;

	float* bd_ax = (float*)calloc(ny + 1, sizeof(float));
	float* bd_bx = (float*)calloc(ny + 1, sizeof(float));
	float* bd_ay = (float*)calloc(nx + 1, sizeof(float));
	float* bd_by = (float*)calloc(nx + 1, sizeof(float));

	MKL_INT ipair[128] = { 0 };
	float* dpair = (float*)calloc((5 * nx / 2 + 7), sizeof(float));

	MKL_INT stat;
	s_init_Helmholtz_2D(&ax, &bx, &ay, &by, &nx, &ny, BCType, &q, ipair, dpair, &stat);
	s_commit_Helmholtz_2D(laplacianResult, bd_ax, bd_bx, bd_ay, bd_by, &xhandle, ipair, dpair, &stat);
	s_Helmholtz_2D(laplacianResult, bd_ax, bd_bx, bd_ay, bd_by, &xhandle, ipair, dpair, &stat);
	free_Helmholtz_2D(&xhandle, ipair, &stat);

	free(bd_ax);
	free(bd_bx);
	free(bd_ay);
	free(bd_by);
	free(dpair);
	return result;
}


cv::Mat HDRCompression::Apply(const cv::Mat& image, cv::Mat& output)
{
	cv::Mat grayImage;
	cv::Mat colorImage;
	cv::Mat yChannel;
	if (image.channels() == 3) {
		colorImage = image.clone();
		colorImage += cv::Scalar(1e-6, 1e-6, 1e-6);
		cv::log(colorImage, colorImage);

		colorImage.convertTo(colorImage, CV_32FC3);
		cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
		cv::cvtColor(image, yChannel, cv::COLOR_BGR2GRAY);

	
	
	}
	else {
		grayImage = image.clone();
		grayImage.convertTo(grayImage, CV_32FC1);
	}
	
	
	std::vector<cv::Mat> pyramid;
	cv::Mat temp = grayImage.clone();
	for (int i = 0; i < HDRCompression::pyramidLevelNeeded; i++) {
		pyramid.push_back(temp);
		cv::pyrDown(temp, temp);

	}
	std::vector<cv::Mat> gradientPyramid;
	for (int i = 0; i < pyramid.size(); i++) {
		cv::Mat gradientMagnitude = CalculateGradient(pyramid[i]);
		gradientMagnitude *= 1.0 / std::pow(2, i + 1);
		gradientPyramid.push_back(std::move(gradientMagnitude));
	}

	float totalIntensityValue = 0;
	float totalPixel = 0;
	for (int i = 0; i < gradientPyramid.size(); i++) {
		totalIntensityValue += cv::sum(gradientPyramid[i])[0];
		totalPixel += gradientPyramid[i].cols * gradientPyramid[i].rows;
	}
	float mean = totalIntensityValue / totalPixel;
	float trueAlpha = mean * HDRCompression::alpha;
	std::vector<cv::Mat> scaleRatePyramid;
	for (int i = 0; i < gradientPyramid.size(); i++) {
		cv::Mat levelScale = CalculateScaleRate(gradientPyramid[i], trueAlpha);

		scaleRatePyramid.push_back(std::move(levelScale));
	}
	cv::Mat attenuation;
	for (int i = scaleRatePyramid.size() - 1; i >= 0; i--) {
		if (i == scaleRatePyramid.size() - 1) {
			attenuation = scaleRatePyramid[i];

			continue;
		}

		cv::resize(attenuation, attenuation, scaleRatePyramid[i].size(), 1, 1);
		attenuation = attenuation.mul(scaleRatePyramid[i]);

	}

	attenuation += 1e-6; // prevent alter direction of gradient

	cv::Mat kernelX = (cv::Mat_<float>(1, 3) << 0, -1, 1);
	cv::Mat kernelY = (cv::Mat_<float>(3, 1) << 0, -1, 1);



	cv::Mat gradientX, gradientY;
	cv::filter2D(grayImage, gradientX, CV_32F, kernelX, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);
	cv::filter2D(grayImage, gradientY, CV_32F, kernelY, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);

	gradientX = gradientX.mul(attenuation);
	gradientY = gradientY.mul(attenuation);

	cv::Mat laplacianX, laplacianY;

	cv::Mat kernelLaplacianX = (cv::Mat_<float>(1, 3) << -1, 1, 0);
	cv::Mat kernelLaplacianY = (cv::Mat_<float>(3, 1) << -1, 1, 0);


	cv::filter2D(gradientX, laplacianX, CV_32F, kernelLaplacianX, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
	cv::filter2D(gradientY, laplacianY, CV_32F, kernelLaplacianY, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

	for (int i = 0; i < laplacianX.rows; i++) {

		laplacianX.at<float>(i, 0) = 2 * gradientX.at<float>(i, 0);

	}

	for (int i = 0; i < laplacianY.cols; i++) {

		laplacianY.at<float>(0, i) = 2 * gradientY.at<float>(0, i);

	}
	cv::Mat finalNewLaplacian = laplacianX + laplacianY;

	cv::Mat resultLog = PoissonSolver(finalNewLaplacian);
	if (image.channels() == 3) {
		cv::Mat result;
		double minVal, maxVal;
		cv::minMaxLoc(resultLog, &minVal, &maxVal); 
		resultLog -= minVal; 
		resultLog = resultLog * (255.0 / (maxVal - minVal));
		cv::normalize(resultLog, resultLog, 1, 255.0, cv::NORM_MINMAX);
		HDRCompression::Color(image, yChannel, resultLog, result);
		result.convertTo(result, CV_8UC3);
		return result;
	}
	cv::normalize(resultLog, resultLog, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	return resultLog;
}

cv::Mat HDRCompression::CalculateGradient(const cv::Mat& image)
{
	cv::Mat kernelX = (cv::Mat_<float>(1, 3) << 0, -1, 1);
	cv::Mat kernelY = (cv::Mat_<float>(3, 1) << 0, -1, 1);

	cv::Mat gradientX, gradientY;
	cv::filter2D(image, gradientX, CV_32F, kernelX, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
	cv::filter2D(image, gradientY, CV_32F, kernelY, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);

	cv::Mat magnitude;
	cv::magnitude(gradientX, gradientY, magnitude);

	return magnitude;
}

cv::Mat HDRCompression::CalculateScaleRate(const cv::Mat& gradientMagnitude, float trueAlpha)
{
	// prevent 0^-1
	gradientMagnitude += 1e-10;
	cv::Mat scaleFactor(gradientMagnitude.size(), CV_32FC1, cv::Scalar(0));
	for (int i = 0; i < gradientMagnitude.rows; i++) {
		for (int j = 0; j < gradientMagnitude.cols; j++) {
			scaleFactor.at<float>(i, j) = std::pow((gradientMagnitude.at<float>(i, j) / trueAlpha), HDRCompression::beta - 1);
		}
	}
	return scaleFactor;
}

cv::Mat HDRCompression::Color(const cv::Mat& originImage, const cv::Mat& ychannel, const cv::Mat& newIntensity, cv::Mat& output)
{
	cv::Mat grayImage = ychannel.clone();
	grayImage += 1e-10;
	cv::Mat grayImage3Channel;
	cv::Mat newIntensity3Channel;
	cv::Mat colorRatio;

	cv::cvtColor(grayImage, grayImage3Channel, cv::COLOR_GRAY2BGR);
	cv::cvtColor(newIntensity, newIntensity3Channel, cv::COLOR_GRAY2BGR);
	cv::divide(originImage, grayImage3Channel, colorRatio);
	cv::Mat colorFactor;
	cv::pow(colorRatio, HDRCompression::staturation, colorFactor);
	output = colorFactor.mul(newIntensity3Channel);
	return output;
}
