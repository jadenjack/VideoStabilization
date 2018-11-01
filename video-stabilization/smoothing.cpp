#include "smoothing.h"

using namespace cv;

void mycv::warpAffine(InputArray src, OutputArray dst,
	InputArray M, Size dsize,
	int flags, int borderMode,
	const Scalar& borderValue) {

	cv::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue);
}