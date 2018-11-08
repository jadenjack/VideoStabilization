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

	const int height = prevImg.rows;
	const int width = prevImg.cols;
	const int frame_len = height * width;
	const unsigned char* prev = prevImg.data;
	const unsigned char* next = nextImg.data;

	int* Ix = new int[frame_len]();
	int* Iy = new int[frame_len]();
	int* It = new int[frame_len]();

	for (int y = 1; y < height; y++) {
		for (int x = 1; x < width; x++) {
			const int idx = y * width + x;
			const int prevDx = (int)prev[idx + 1] - (int)prev[idx - 1];
			const int nextDx = (int)next[idx + 1] - (int)next[idx - 1];
			const int prevDy = (int)prev[idx + width] - (int)prev[idx - width];
			const int nextDy = (int)next[idx + width] - (int)next[idx - width];

			Ix[idx] = (prevDx + nextDx) * 0.5;
			Iy[idx] = (prevDy + nextDy) * 0.5;
			It[idx] = (int)next[idx] - (int)prev[idx];
		}
	}

	for (int p = 0; p < prevPts.size(); p++) {
		const Point2f pt = prevPts[p];
		float sumxx = 0;
		float sumxy = 0;
		float sumyy = 0;
		float sumxt = 0;
		float sumyt = 0;
		bool isMatched = true;

		for (int wy = -winSize.height / 2; wy <= winSize.height / 2; wy++) {
			for (int wx = -winSize.width / 2; wx <= winSize.width / 2; wx++) {
				const int x = pt.x + wx;
				const int y = pt.y + wy;
				const int idx = y * width + x;

				if (idx < 0 || idx >= frame_len) {
					isMatched = false;
					break;
				}
				sumxx += Ix[idx] * Ix[idx];
				sumxy += Ix[idx] * Iy[idx];
				sumyy += Iy[idx] * Iy[idx];
				sumxt += Ix[idx] * It[idx];
				sumyt += Iy[idx] * It[idx];
			}
		}

		const float G[2][2] = {
			{sumxx, sumxy},
			{sumxy, sumyy}
		};

		const float det = G[0][0] * G[1][1] - G[0][1] * G[1][0];
		if (!isMatched || (det) < criteria.epsilon) {
			nextPts.push_back(pt);
			status.push_back(0);
			continue;
		}

		const float vx = (-sumyy * sumxt + sumxy * sumyt) / det;
		const float vy = (-sumxy * sumxt - sumxx * sumyt) / det;

		nextPts.push_back(pt + Point2f(vx, vy));
		status.push_back(1);
	}

	delete[] Ix;
	delete[] Iy;
	delete[] It;
}