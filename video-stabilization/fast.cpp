#include "fast.h"

using namespace cv;

void mycv::goodFeaturesToTrack(Mat image, vector<Point2f>& corners,
	int maxCorners, double qualityLevel, double minDistance,
	InputArray mask, int blockSize, bool useHarrisDetector, double k) {

	cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance,
		mask, blockSize, useHarrisDetector, k);
}