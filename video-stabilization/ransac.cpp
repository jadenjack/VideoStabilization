#include "ransac.h"

using namespace cv;

Mat mycv::estimateRigidTransform(vector<Point2f> src, vector<Point2f> dst, bool fullAffine) {
	return cv::estimateRigidTransform(src, dst, fullAffine);
}