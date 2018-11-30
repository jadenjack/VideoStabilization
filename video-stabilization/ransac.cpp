#include "ransac.h"

using namespace cv;

//Mat mycv::estimateRigidTransform(vector<Point2f> src, vector<Point2f> dst, bool fullAffine) {
//
//	return cv::estimateRigidTransform(src, dst, fullAffine);
//}

void plot(const vector<Point2f>& previousFeaturesIn, const vector<Point2f>& currentFeaturesIn,
	const int* inliersIndices, int inliersCount,
	Mat xCoef, Mat yCoef);

Mat mycv::estimateRigidTransform(vector<Point2f> src, vector<Point2f> dst, bool fullAffine) {
	
	assert(src.size() == dst.size());

	const int pairCount = src.size();
	Mat srcMat(pairCount * 2, 4, CV_32FC1);
	Mat dstVec(pairCount * 2, 1, CV_32FC1);

	Mat* srcMatArray = new Mat[pairCount];
	Mat* dstVecArray = new Mat[pairCount];
	for (int i = 0; i < pairCount; ++i) {
		srcMatArray[i] = (Mat_<float>(2, 4) 
			<< src.at(i).x, -src.at(i).y,  1,  0,
			   src.at(i).y,  src.at(i).x,  0,  1
			);

		dstVecArray[i] = (Mat_<float>(2, 1)
			<< dst.at(i).x, 
			   dst.at(i).y
			);
	}
	vconcat(srcMatArray, pairCount, srcMat);
	vconcat(dstVecArray, pairCount, dstVec);

	Mat inverted(srcMat.cols, srcMat.rows, CV_32FC1);
	invert(srcMat, inverted, DECOMP_SVD);
	Mat coefMat = inverted * dstVec;

	const float a = coefMat.at<float>(0, 0);
	const float b = coefMat.at<float>(1, 0);
	const float c = coefMat.at<float>(2, 0);
	const float d = coefMat.at<float>(3, 0);
	const float s = sqrt(a*a + b * b);
	const float coss = a / s;
	const float sinn = b / s;

	Mat transform = (Mat_<double>(2, 3) 
		<< coss, -sinn,  c,
		   sinn,  coss,  d
		);

	return transform.clone();
}

void mycv::extractInliers(const vector<Point2f>& previousFeaturesIn, const vector<Point2f>& currentFeaturesIn,
						  vector<Point2f>* previousInliersOut, vector<Point2f>* currentInliersOut) {
	const int MAX_ITERATION = 150;
	const int orderOfPolynomials = 2;
	const int sampleCount = orderOfPolynomials+1;
	const int dataCount = previousFeaturesIn.size();

	// calculate good threshold
	float th = 3;
#pragma region 
	/*float xMean = 0.0;
	float yMean = 0.0;
	int pointsCount = src.size();


	for (int i = 0; i < pointsCount; ++i) {
		xMean += src[i].x;
		yMean += src[i].y;
	}
	xMean /= pointsCount;
	yMean /= pointsCount;

	float xVariance = 0.0;
	float yVariance = 0.0;

	for (int i = 0; i < pointsCount; ++i) {
		xVariance += (src[i].x - xMean)*(src[i].x - xMean);
		yVariance += (src[i].y - yMean)*(src[i].y - yMean);
	}

	const float xDev = sqrt(xVariance);
	const float yDev = sqrt(yVariance);

	*xThreshold = 2 * xDev;
	*yThreshold = 2 * yDev;*/
#pragma endregion

	int inliersCount = 0;
	Mat finalXCoef(3, 1, CV_64F);
	Mat finalYCoef(3, 1, CV_64F);
	
	int *bestInliers = new int[dataCount];
	int *testInliers = new int[dataCount];
	

	int iter = 0;
	int maxIter = 9999999;
	while( ++iter && iter < maxIter && iter < MAX_ITERATION) {
		Mat inX(3, 3, CV_64F);
		Mat outX(3, 1, CV_64F);

		Mat inY(3, 3, CV_64F);
		Mat outY(3, 1, CV_64F);

		//
		int index[] = { rand() % dataCount, rand() % dataCount, rand() % dataCount };
		for (int i = 0; i < 3; ++i) {
			double xRow[] = { pow(currentFeaturesIn[index[i]].x,2), currentFeaturesIn[index[i]].x, 1 };
			double yRow[] = { pow(currentFeaturesIn[index[i]].y,2), currentFeaturesIn[index[i]].y, 1 };
			for (int j = 0; j < 3; ++j) {
				inX.at<double>(i, j) = xRow[j];
				inY.at<double>(i, j) = yRow[j];
			}
			outX.at<double>(i, 0) = previousFeaturesIn[index[i]].x;
			outY.at<double>(i, 0) = previousFeaturesIn[index[i]].y;
		}

		invert(inX, inX, DECOMP_SVD);
		invert(inY, inY, DECOMP_SVD);

		Mat xCoef = inX * outX;
		Mat yCoef = inY * outY;

		int count = 0;
		for (int i = 0; i < dataCount; ++i) {
			double x = currentFeaturesIn[i].x;
			double y = currentFeaturesIn[i].y;
			double fHatx = x * x*xCoef.at<double>(0, 0) + x * xCoef.at<double>(1, 0) + xCoef.at<double>(2, 0);
			double fHaty = y * y*yCoef.at<double>(0, 0) + y * yCoef.at<double>(1, 0) + yCoef.at<double>(2, 0);
			double fx = previousFeaturesIn[i].x;
			double fy = previousFeaturesIn[i].y;
			if (abs(fx - fHatx) < th
				&& abs(fy - fHaty) < th) {
				testInliers[count] = i;
				++count;
			}
		}

		if (count > inliersCount) {
			finalXCoef = xCoef;
			finalYCoef = yCoef;
			inliersCount = count;

			int* temp = bestInliers;
			bestInliers = testInliers;
			testInliers = temp;

			float w = (float)inliersCount / (float)dataCount;
			maxIter = log(0.01) / log(1 - pow(w, sampleCount));
		}
	}
	
	for (int i = 0; i < inliersCount; ++i) {
		previousInliersOut->push_back(previousFeaturesIn[bestInliers[i]]);
		currentInliersOut->push_back(currentFeaturesIn[bestInliers[i]]);
	}
	cout << "#iteration: " << iter << " / "<<dataCount << "->" << inliersCount << endl;
	//plot(previousFeaturesIn, currentFeaturesIn, bestInliers, inliersCount, finalXCoef, finalYCoef);

	delete bestInliers;
	delete testInliers;
}

void mycv::plot(const vector<Point2f>& previousFeaturesIn, const vector<Point2f>& currentFeaturesIn,
		const int* inliersIndices, int inliersCount,
		Mat xCoef, Mat yCoef) {
	const int dataCount = previousFeaturesIn.size();
	const int frameSize = 500;
	double maxX = 0;
	double maxY = 0;
	for (int i = 0; i < dataCount; ++i) {
		if (currentFeaturesIn[i].x > maxX) { maxX = currentFeaturesIn[i].x; }
		if (previousFeaturesIn[i].x > maxX) { maxX = previousFeaturesIn[i].x; }
		if (currentFeaturesIn[i].y > maxY) { maxY = currentFeaturesIn[i].y; }
		if (previousFeaturesIn[i].y > maxY) { maxY = previousFeaturesIn[i].y; }
	}

	double xNormalizeFactor = (frameSize - 20) / maxX;
	double yNormalizeFactor = (frameSize - 20) / maxY;
	Mat px(frameSize, frameSize, CV_8UC3);
	Mat py(frameSize, frameSize, CV_8UC3);
	px = Scalar::all(255);
	py = Scalar::all(255);

	int nextInlier = 0;
	const Vec3b inlierColor = { 0, 0, 255 };
	const Vec3b outlierColor = { 0, 0, 0 };

	for (int i = 0; i < dataCount; ++i) {
		int u1 = currentFeaturesIn[i].x*xNormalizeFactor + 10;
		int v1 = previousFeaturesIn[i].x*xNormalizeFactor + 10;

		int u2 = currentFeaturesIn[i].y*yNormalizeFactor + 10;
		int v2 = previousFeaturesIn[i].y*yNormalizeFactor + 10;

		Vec3b color;
		if (nextInlier < inliersCount && inliersIndices[nextInlier] == i) {
			color = inlierColor;
			++nextInlier;
		}
		else {
			color = outlierColor;
		}

		if (u1 >= 1 && v1 >= 1) {
			px.at <Vec3b>(u1, v1) = color;
			px.at <Vec3b>(u1 - 1, v1) = color;
			px.at <Vec3b>(u1, v1 - 1) = color;
			px.at <Vec3b>(u1 + 1, v1) = color;
			px.at <Vec3b>(u1, v1 + 1) = color;
		}

		if (u2 >= 1 && v2 >= 1) {
			py.at <Vec3b>(u2, v2) = color;
			py.at <Vec3b>(u2 - 1, v2) = color;
			py.at <Vec3b>(u2, v2 - 1) = color;
			py.at <Vec3b>(u2 + 1, v2) = color;
			py.at <Vec3b>(u2, v2 + 1) = color;
		}
	}

	for (int i = 0; i < maxX; ++i) {
		double fx = i * i*xCoef.at<double>(0, 0) + i * xCoef.at<double>(1, 0) + xCoef.at<double>(2, 0);
		if (fx < 0)continue;
		double x = i * xNormalizeFactor;
		double y = fx * xNormalizeFactor;
		if (y < 0 || y>499) continue;
		px.at<Vec3b>(x, y) = outlierColor;
	}

	for (int i = 0; i < maxY; ++i) {
		double fx = i * i*yCoef.at<double>(0, 0) + i * yCoef.at<double>(1, 0) + yCoef.at<double>(2, 0);
		if (fx < 0)continue;
		double x = i * yNormalizeFactor;
		double y = fx * yNormalizeFactor;
		if (y < 0 || y>499) continue;
		py.at<Vec3b>(x, y) = outlierColor;
	}

	imshow("x", px);
	imshow("y", py);
	waitKey();
}