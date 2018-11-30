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

	const int width = prevImg.cols;
	const int height = prevImg.rows;
	const int frameSize = width * height;
	const unsigned char* prevData = prevImg.data;
	const unsigned char* nextData = nextImg.data;

	vector<double> Ix(frameSize);
	vector<double> Iy(frameSize);

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			const int idx = y * width + x;
			Ix[idx] = 0.5 * ((int)prevData[idx + 1] - (int)prevData[idx - 1]);
			Iy[idx] = 0.5 * ((int)prevData[idx + width] - (int)prevData[idx - width]);
		}
	}

	for (int p = 0; p < prevPts.size(); p++) {
		const Point2f from = prevPts[p];
		Point2f optFlowVector(0, 0);
		double G[2][2] = {
			{0, 0},
			{0, 0}
		};

		for (int wy = -winSize.height; wy <= winSize.height; wy++) {
			for (int wx = -winSize.width; wx <= winSize.width; wx++) {
				const int y = from.y + wy;
				const int x = from.x + wx;

				if (x < 0 || width <= x || y < 0 || height <= y) {
					continue;
				}

				const int idx = y * width + x;
				G[0][0] += Ix[idx] * Ix[idx];
				G[0][1] += Ix[idx] * Iy[idx];
				G[1][0] += Ix[idx] * Iy[idx];
				G[1][1] += Iy[idx] * Iy[idx];
			}
		}

		const double det = G[0][0] * G[1][1] - G[0][1] * G[1][0];
		const double InvG[2][2] = {
			{G[1][1] / det, -G[0][1] / det},
			{-G[1][0] / det, G[0][0] / det}
		};

		for (int i = 0; i < criteria.maxCount; i++) {
			Point2f mismatch(0, 0);

			for (int wy = -winSize.height; wy <= winSize.height; wy++) {
				for (int wx = -winSize.width; wx <= winSize.width; wx++) {
					const int fy = from.y + wy;
					const int fx = from.x + wx;
					const int ty = from.y + wy + optFlowVector.y;
					const int tx = from.x + wx + optFlowVector.x;

					if (fx < 0 || width <= fx || fy < 0 || height <= fy
						|| tx < 0 || width <= tx || ty < 0 || height <= ty) {
						continue;
					}

					const int diff = (int)prevData[fy * width + fx] - (int)nextData[ty * width + tx];
					mismatch.y += diff * Iy[fy * width + fx];
					mismatch.x += diff * Ix[fy * width + fx];
				}
			}

			const Point2f gap(InvG[0][0] * mismatch.x + InvG[0][1] * mismatch.y, InvG[1][0] * mismatch.x + InvG[0][1] * mismatch.y);
			optFlowVector += gap;
			
			const double norm = sqrt(gap.x * gap.x + gap.y * gap.y);
			if (norm < criteria.epsilon) {
				break;
			}
		}

		const Point2f to = from + optFlowVector;
		const bool isInsideFrame = 0 <= to.x && to.x < width && 0 <= to.y && to.y < height;
		nextPts.push_back(to);
		status.push_back(isInsideFrame);
	}
}