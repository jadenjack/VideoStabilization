#include "smoothing.h"

using namespace cv;

void mycv::warpAffine(Mat src, Mat& dst,
	Mat M, Size dsize,
	int flags, int borderMode,
	const Scalar& borderValue) {

	cv::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue);
}