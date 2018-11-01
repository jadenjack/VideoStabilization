#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

const int SMOOTHING_RADIUS = 30; // In frames. The larger the more stable the video, but less reactive to sudden panning
const int HORIZONTAL_BORDER_CROP = 20; // In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.

namespace mycv {
	struct TransformParam
	{
		TransformParam() {}
		TransformParam(double _dx, double _dy, double _da) {
			dx = _dx;
			dy = _dy;
			da = _da;
		}

		double dx;
		double dy;
		double da; // angle
	};

	struct Trajectory
	{
		Trajectory() {}
		Trajectory(double _x, double _y, double _a) {
			x = _x;
			y = _y;
			a = _a;
		}

		double x;
		double y;
		double a; // angle
	};

	void goodFeaturesToTrack(InputArray image, OutputArray corners,
		int maxCorners, double qualityLevel, double minDistance,
		InputArray mask = noArray(), int blockSize = 3,
		bool useHarrisDetector = false, double k = 0.04);

	void calcOpticalFlowPyrLK(InputArray prevImg, InputArray nextImg,
		InputArray prevPts, InputOutputArray nextPts,
		OutputArray status, OutputArray err,
		Size winSize = Size(21, 21), int maxLevel = 3,
		TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01),
		int flags = 0, double minEigThreshold = 1e-4);

	Mat estimateRigidTransform(InputArray src, InputArray dst, bool fullAffine);

	void warpAffine(InputArray src, OutputArray dst,
		InputArray M, Size dsize,
		int flags = INTER_LINEAR,
		int borderMode = BORDER_CONSTANT,
		const Scalar& borderValue = Scalar());

	void cvtColor(InputArray src, OutputArray dst, int code, int dstCn = 0);

	void resize(InputArray src, OutputArray dst,
		Size dsize, double fx = 0, double fy = 0,
		int interpolation = INTER_LINEAR);
}
