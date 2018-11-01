#include "sift.h"

using namespace cv;

void mycv::calcOpticalFlowPyrLK(InputArray prevImg, InputArray nextImg,
	InputArray prevPts, InputOutputArray nextPts,
	OutputArray status, OutputArray err,
	Size winSize, int maxLevel,
	TermCriteria criteria, int flags, double minEigThreshold) {

	cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err,
		winSize, maxLevel, criteria, flags, minEigThreshold);
}