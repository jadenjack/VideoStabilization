#pragma once

#include <opencv2/opencv.hpp>
#include "mystab.h"
namespace mycv {
	void calcOpticalFlowLK(Mat prevImg, Mat nextImg,
		const vector<double>& Ix, const vector<double>& Iy,
		const Point2f from, const Point2f guess, Point2f& optFlowVector,
		Size winSize, TermCriteria criteria);
	
	int calcMaximumLevel(const Mat img, const Size winSize, const int hopeThisLevel);

	Mat resizeForPyramid(const Mat src, const int maxLevel);

	Mat makeNextLevelImage(const Mat src);
	
	void calcDerivatives(const Mat img, vector<double>& Ix, vector<double>& Iy);
}
