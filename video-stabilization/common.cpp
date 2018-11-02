#include <opencv2/opencv.hpp>
#include "mystab.h"

using namespace cv;

void mycv::cvtColor(InputArray src, OutputArray dst, int code, int dstCn) {
	cv::cvtColor(src, dst, code, dstCn);
}

void mycv::resize(InputArray src, OutputArray dst,
	Size dsize, double fx, double fy,
	int interpolation) {

	cv::resize(src, dst, dsize, fx, fy, interpolation);
}
