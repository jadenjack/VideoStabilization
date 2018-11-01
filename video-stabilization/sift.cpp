#include <iostream>
#include "sift.h"

using namespace cv;
using namespace std;

void mycv::calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg,
	vector<Point2f> prevPts, vector<Point2f>& nextPts,
	vector<uchar>& status, vector<float>& err,
	Size winSize, int maxLevel,
	TermCriteria criteria, int flags, double minEigThreshold) {

	cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err,
		winSize, maxLevel, criteria, flags, minEigThreshold);
}