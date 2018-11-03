#pragma once

#include <opencv2/opencv.hpp>
#include "mystab.h"

#define PAD 2
#define FILTER_SIZE (2*PAD+1)

const int lowpass_filter[FILTER_SIZE][FILTER_SIZE] = {
	{1,  4,  6,  4, 1},
	{4, 16, 24, 16, 4},
	{6, 24, 36, 24, 6},
	{4, 16, 24, 16, 4},
	{1,  4,  6,  4, 1},
};

// Lowpass filter를 적용 한 다음 피라미드 이미지 리턴
Mat nextFloor(const Mat img);