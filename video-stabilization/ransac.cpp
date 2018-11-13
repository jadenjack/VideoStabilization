#include "ransac.h"

using namespace cv;

//Mat mycv::estimateRigidTransform(vector<Point2f> src, vector<Point2f> dst, bool fullAffine) {
//
//	return cv::estimateRigidTransform(src, dst, fullAffine);
//}

Mat mycv::estimateRigidTransform(vector<Point2f> src, vector<Point2f> dst, bool fullAffine) {
	if (src.size() != dst.size()) {
		return Mat::zeros(Size(3, 3), CV_32FC1);
	}

	int pairCount = src.size();

	Mat prevMat(pairCount << 1, 4, CV_32FC1);
	Mat cvtVec(pairCount << 1, 1, CV_32FC1);
	std::cout << pairCount << std::endl;
	int matIndex = 0;
	int vecIndex = 0;

	// 모든 쌍의 정보를 행렬에 추가
	for (int i = 0; i < pairCount; ++i) {
		float inX = src.at(i).x;
		float inY = src.at(i).y;
		float outX = dst.at(i).x;
		float outY = dst.at(i).y;

		float addMat[] = {
			inX, -inY, 1, 0,
			inY, inX, 0, 1
		};
		float addVec[] = {
			outX,
			outY
		};

		int addMatSize = sizeof(addMat);
		memcpy(&prevMat.data[matIndex], addMat, addMatSize);
		matIndex += addMatSize;

		int addVecSize = sizeof(float) << 1;
		memcpy(&cvtVec.data[vecIndex], addVec, addVecSize);
		vecIndex += addVecSize;
	}

	Mat inverted(prevMat.cols, prevMat.rows, CV_32FC1);
	invert(prevMat, inverted, DECOMP_SVD);
	Mat coefMat = inverted * cvtVec;

	float a = coefMat.at<float>(0, 0);
	float b = coefMat.at<float>(1, 0);
	float c = coefMat.at<float>(2, 0);
	float d = coefMat.at<float>(3, 0);
	float s = sqrt(a*a + b * b);
	float coss = a / s;
	float sinn = b / s;

	double tElements[] = {
		coss, -sinn, c,
		sinn, coss, d
	};

	Mat transform(2, 3, CV_64FC1, tElements);

	return transform.clone();
}