#include "ransac.h"

using namespace cv;

Mat mycv::estimateRigidTransform(InputArray src, InputArray dst, bool fullAffine) {
	return cv::estimateRigidTransform(src, dst, fullAffine);
}