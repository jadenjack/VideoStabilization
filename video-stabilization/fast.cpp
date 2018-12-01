#include "fast.h"

using namespace cv;

/*
Sobel을 적용하여
dx, dy추출
->Mat dx2, dy2, dxy 구함
->gaussian filter적용
->이것들을 갖고 M을 만들어냄
->EigenValues생성
->nonMaximumSuppression
->기준치 이하인 코너들은 제거
->min distance
*/

//2 x 2 Matrix
//                         [ dx^2  - λ , dx*dy    ]
// Gaussian Kernel      *  [ dx*dy     , dy^2 - λ ] 
//고유값(EigenValue)를 구하기 위해서는 위 행렬의 Determinant값이 0이되어야 한다.
//http://darkpgmr.tistory.com/105

void mycv::goodFeaturesToTrack(Mat src, vector<Point2f>& corners,
	int maxCorners, double qualityLevel, double minDistance, InputArray mask, int blockSize, bool useHarrisDetector, double k) {

	//cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);

	blockSize = 3;
	Mat dx, dy;
	vector<float> quality;
	int padding = blockSize / 2;
	float maxQuality = 0;
	Mat minEigenValues;
	minEigenValues = Mat::zeros(src.rows, src.cols, CV_32FC1);
	
	
	vector<qualityPosition> qualityMap = {};
	Mat gaussianKernel = get3by3GaussianKernel();

	//get minimum Eigen Values of every single pixel
	sobelDerivative(src, &dx, &dy, blockSize, padding);
	applyFilterToDerivative(&dx, &dy, gaussianKernel);
	getMinimumEigenValues(&minEigenValues, dx, dy);
	nonMaximumSuppression(&minEigenValues,&qualityMap, blockSize, padding, &maxQuality);
	rejectLowQuality(&qualityMap, &maxQuality, qualityLevel);
	sortByQuality(&qualityMap);
	applyMinDistance(src.rows, src.cols, &corners, &qualityMap, &maxQuality, maxCorners, qualityLevel);
}


void mycv::sobelDerivative(Mat src, Mat* dx, Mat* dy, int blockSize, int padding) {

	int r, c;
	*dx = Mat::zeros(src.rows, src.cols, CV_32FC1);
	*dy = Mat::zeros(src.rows, src.cols, CV_32FC1);
	for (r = padding; r < src.rows - padding; r++) {
		for (c = padding; c < src.cols - padding; c++) {
			calculateDerivative(src, dx, dy, r, c);
		}
	}

}

//blockSize = 3 기준
void mycv::calculateDerivative(Mat src, Mat* dx, Mat* dy, int i, int j) {
	int sobelDx, sobelDy;
	// -1 0 1
	// -2 0 2
	// -1 0 1
	sobelDx = src.data[(i - 1)*src.rows + j + 1] + src.data[(i)*src.rows + j + 1] * 2 + src.data[(i + 1)*src.rows + j + 1]
		- src.data[(i - 1)*src.rows + j - 1] - src.data[(i)*src.rows + j - 1] * 2 - src.data[(i + 1)*src.rows + j - 1];
	// -1 -2 -1
	//  0  0  0
	//  1  2  1
	sobelDy = -src.data[(i - 1)*src.rows + j - 1] - src.data[(i - 1)*src.rows + j] * 2 - src.data[(i - 1)*src.rows + j + 1]
		+ src.data[(i + 1)*src.rows + j - 1] + src.data[(i + 1)*src.rows + j] * 2 + src.data[(i + 1)*src.rows + j + 1];
	dx->at<float>(i, j) = sobelDx/65536.0;
	dy->at<float>(i, j) = sobelDy / 65536.0;
}

void mycv::applyFilterToDerivative(Mat* dx, Mat* dy, Mat gaussianKernel) {
	*dx = applyGaussianKernel(*dx, gaussianKernel);
	*dy = applyGaussianKernel(*dy, gaussianKernel);
}

Mat mycv::applyGaussianKernel(Mat& d, Mat kernel) {

	Mat filtered = Mat::zeros(d.rows, d.cols, CV_32FC1);
	int r, c;
	int i, j;
	float sum;
	for (r = 1; r < d.rows - 1; r++) {
		for (c = 1; c < d.cols - 1; c++) {

			sum = 0;
			//apply kernel
			for (i = r - 1; i <= r + 1; i++) {
				for (j = c - 1; j <= c + 1; j++) {
					sum += d.at<float>(i, j) * kernel.at<float>(i - r + 1, j - c + 1);
				}
			}
			filtered.at<float>(r, c) = sum;
		}
	}
	return filtered;
}

Mat mycv::get3by3GaussianKernel() {
	Mat kernel = Mat::zeros(3, 3, CV_32FC1);
	kernel.at<float>(0,0) = 1.0 / 16;
	kernel.at<float>(0, 1) = 1.0 / 8;
	kernel.at<float>(0, 2) = 1.0/ 16;
	kernel.at<float>(1, 0) = 1.0 / 8;
	kernel.at<float>(1, 1) = 1.0 / 4;
	kernel.at<float>(1, 2) = 1.0 / 8;
	kernel.at<float>(2, 0) = 1.0 / 16;
	kernel.at<float>(2, 1) = 1.0 / 8;
	kernel.at<float>(2, 2) = 1.0 / 16;
	return kernel;
}

//minimum EigenValue는 근의 공식 중 작은 값이다.
//https://www.cs.cmu.edu/~ggordon/imageproc/edges.html
//[ a  b ]
//[ b  d ] 행렬에서 eigenValue 구하기.

//a+d + sqrt(pow(a-d,2) + 4*b*b)
void mycv::getMinimumEigenValues(Mat* minEigenValues, Mat dx, Mat dy) {

	int r, c;
	Mat a = Mat::zeros(minEigenValues->rows, minEigenValues->cols, CV_32FC1);
	Mat b = Mat::zeros(minEigenValues->rows, minEigenValues->cols, CV_32FC1);
	Mat d = Mat::zeros(minEigenValues->rows, minEigenValues->cols, CV_32FC1);
	Mat left = Mat::zeros(minEigenValues->rows, minEigenValues->cols, CV_32FC1);
	Mat right = Mat::zeros(minEigenValues->rows, minEigenValues->cols, CV_32FC1);;
	Mat right_left = Mat::zeros(minEigenValues->rows, minEigenValues->cols, CV_32FC1);;
	Mat right_right = Mat::zeros(minEigenValues->rows, minEigenValues->cols, CV_32FC1);;

	multiply(dx, dx, a);
	multiply(dx, dy, b);
	multiply(dy, dy, d);
	add(a, d, left);
	pow(a - d, 2, right_left);
	multiply(b, b, right_right);
	multiply(right_right, 4, right_right);
	sqrt(right_right+right_right, right);
	subtract(left, right, *minEigenValues);
}



void mycv::nonMaximumSuppression(Mat* minEigenValues, vector<qualityPosition>* qualityMap, int blockSize, int padding, float* maxQuality) {

	int r, c;
	int vectorIndex = 0;
	for (r = padding; r < minEigenValues->rows - padding; r++) {
		for (c = padding; c < minEigenValues->cols - padding; c++) {

			if (!maximumValue(*minEigenValues, r, c, padding))
				minEigenValues->at<float>(r, c) = 0;
			else {

				qualityMap->push_back(qualityPosition(r, c, minEigenValues->at<float>(r, c)));
				//get Max Quality
				if (*maxQuality < minEigenValues->at<float>(r, c)) {
					*maxQuality = minEigenValues->at<float>(r, c);
				}
			}
		}
	}

}

bool mycv::maximumValue(Mat minEigenValues, int r, int c, int padding) {

	float target = minEigenValues.at<float>(r, c);
	for (int i = r - padding; i <= r + padding; i++) {
		for (int j = c - padding; j <= c + padding; j++) {
			if (minEigenValues.at<float>(i, j) > target)
				return false;
		}
	}
	return true;
}

void mycv::rejectLowQuality(vector<qualityPosition>* qualityMap, float* maxQuality, float qualityLevel) {
	vector<qualityPosition>::iterator iter = qualityMap->begin();
	float boundary = (*maxQuality) * qualityLevel;
	while (iter != qualityMap->end()) {
		float eigenValue = iter->q;
		if (eigenValue < boundary) {
			iter = qualityMap->erase(iter);
		}
		else {
			iter++;
		}
	}

}

void mycv::sortByQuality(vector<qualityPosition>* qualityMap) {
	sort(qualityMap->begin(), qualityMap->end(), greaterQuality());
}

void mycv::applyMinDistance(int rows, int cols, vector<Point2f>* corners, vector<qualityPosition>* qualityMap, float* maxQuality,int maxCorners, float qualityLevel) {
	int cornerCount = 0;
	int d = 5;
	vector<qualityPosition>::iterator iter = qualityMap->begin();
	float boundary = (*maxQuality) * qualityLevel;
	//&& cornerCount<maxCorners
	
	vector<vector<bool>> strongCorner(rows, vector<bool>(cols,true));
	while (iter != qualityMap->end() && cornerCount<maxCorners) {
		int r = (*iter).r;
		int c = (*iter).c;
		iter++;
		if (strongCorner[r][c] == false) {
			continue;
		}

		corners->push_back(Point2f(r,c));
		cornerCount++;

		for (int i = -d; i <= d; i++) {
			for (int j = -d; j <= d; j++) {
				if (r + i < 0 || c + j < 0 || r + i >= rows || c + j >= cols)
					continue;

				if (strongCorner[r+i][c+j] == true) {
					strongCorner[r+i][c+j] = false;
				}
			}
		}

	}
}