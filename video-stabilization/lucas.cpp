#include <iostream>
#include <algorithm>
#include "lucas.h"

using namespace cv;
using namespace std;

void mycv::calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg,
	vector<Point2f> prevPts, vector<Point2f>& nextPts,
	vector<uchar>& status, vector<float>& err,
	Size winSize, int maxLevel,
	TermCriteria criteria, int flags, double minEigThreshold) {
	
	vector<Mat> prevImgs, nextImgs;
	prevImgs.push_back(prevImg);
	nextImgs.push_back(nextImg);

	for (int i = 1; i <= maxLevel; i++) {
		prevImgs.push_back(nextFloor(prevImgs[i - 1]));
		nextImgs.push_back(nextFloor(nextImgs[i - 1]));
	}

	for (int p = 0; p < prevPts.size(); p++) {
		vector<Point2f> guesses(maxLevel+1, Point2f(0, 0));
		const Point2f prevPt = prevPts[p];
		
		for (int L = maxLevel; L >= 0; L--) {
			const Point2f localPrevPt(prevPt / pow(2, L));

		}
	}

	cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold);
}

Mat nextFloor(const Mat img) {
	const int rows = (img.rows / 2) % 2 ? img.rows / 2 - 1 : img.rows / 2;
	const int cols = (img.cols / 2) % 2 ? img.cols / 2 - 1 : img.cols / 2;
	Mat next = Mat::zeros(rows, cols, img.type());

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			int sum = 0;
			int weight = 0;
			for (int py = -PAD; py < PAD; py++) {
				for (int px = -PAD; px < PAD; px++) {
					const int r = 2*y + py;
					const int c = 2*x + px;
					if (r >= 0 && r < img.rows && c >= 0 && c < img.cols) {
						sum += img.at<unsigned char>(r, c) * lowpass_filter[PAD + py][PAD + px];
						weight += lowpass_filter[PAD + py][PAD + px];
					}
				}
			}
			next.at<unsigned char>(y, x) = sum / weight;
		}
	}
	return next;
}
