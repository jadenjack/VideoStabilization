#pragma once

#include <opencv2/opencv.hpp>
#include "mystab.h"

const int dy[] = { -3,-3,-2,-1, 0, 1, 2, 3, 3, 3, 2, 1, 0,-1,-2,-3 };
const int dx[] = { 0, 1, 2, 3, 3, 3, 2, 1, 0,-1,-2,-3,-3,-3,-2,-1 };
const Scalar BLACK(0, 0, 0);
RNG rng(12345);

namespace mycv {

	class qualityPosition{
	public:
		qualityPosition(int row, int col, float quality) {
			r = row;
			c = col;
			q = quality;
		}
		float r;
		float c;
		//quality
		float q;
	};

	struct greaterQuality {
		bool operator()(qualityPosition const &a, qualityPosition const &b) const noexcept {
			return a.q > b.q;
		}
	};

	void sobelDerivative(Mat src, Mat* dx, Mat* dy, int blockSize, int padding);
	void applyFilterToDerivative(Mat* dx, Mat* dy, Mat gaussianKernel);
	Mat applyGaussianKernel(Mat& d, Mat kernel);
	Mat get3by3GaussianKernel();
	void getMinimumEigenValues(Mat* minEigenValues, Mat dx, Mat dy);
	void nonMaximumSuppression(Mat* minEigenValues, vector<qualityPosition>* qualityMap, int blockSize, int padding, float* maxQuality);
	void calculateDerivative(Mat src, Mat* dx, Mat* dy, int i, int j);
	bool maximumValue(Mat minEigenValues, int r, int c, int padding);
	void rejectLowQuality(vector<qualityPosition>* qualityMap, float* maxQuality, float qualityLevel);
	void sortByQuality(vector<qualityPosition>* qualityMap);
	void applyMinDistance(int rows, int cols, vector<Point2f>* corners, vector<qualityPosition>* qualityMap, float* maxQuality,int maxCorners, float qualityLevel);
}