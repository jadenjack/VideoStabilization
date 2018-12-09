/*
Copyright (c) 2014, Nghia Ho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include "mystab.h"

#define KEY_SPACE 0x20
#define KEY_LEFT 0x250000
#define KEY_UP 0x260000
#define KEY_RIGHT 0x270000
#define KEY_DOWN 0x280000

#define MAX_WIDTH 800
#define MAX_HEIGHT 400


using namespace std;
using namespace cv;

// This video stablisation smooths the global trajectory using a sliding average window


// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video
int main(int argc, char **argv)
{
	// For further analysis
	//ofstream out_transform("prev_to_cur_transformation.txt");
	//ofstream out_trajectory("trajectory.txt");
	//ofstream out_smoothed_trajectory("smoothed_trajectory.txt");
	//ofstream out_new_transform("new_prev_to_cur_transformation.txt");
	if (argc < 2) {
		cout << "영상을 입력해주세요." << endl;
		return 0;
	}

	const string inputPath = string(argv[1]);

	VideoCapture cap(inputPath);
	assert(cap.isOpened());

	Mat cur, cur_grey;
	Mat prev, prev_grey;
	Mat opencv_cur, opencv_cur_grey;
	Mat opencv_prev, opencv_prev_grey;

	cap >> prev;
	mycv::cvtColor(prev, prev_grey, COLOR_BGR2GRAY);
	cv::cvtColor(prev, opencv_prev_grey, COLOR_BGR2GRAY);

	// Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
	vector <mycv::TransformParam> prev_to_cur_transform; // previous to current
	vector <mycv::TransformParam> opencv_prev_to_cur_transform;
	int k = 1;
	int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	Mat last_T;
	Mat opencv_last_T;

	while (true) {
		cap >> cur;

		if (cur.data == NULL) {
			break;
		}

		mycv::cvtColor(cur, cur_grey, COLOR_BGR2GRAY);
		cv::cvtColor(cur, opencv_cur_grey, COLOR_BGR2GRAY);
		// vector from prev to cur
		vector <Point2f> prev_corner, cur_corner;
		vector <Point2f> prev_corner2, cur_corner2;
		vector <Point2f> opencv_prev_corner, opencv_cur_corner;
		vector <Point2f> opencv_prev_corner2, opencv_cur_corner2;
		vector <uchar> status;
		vector <uchar> opencv_status;
		vector <float> err;
		vector <float> opencv_err;

		mycv::goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
		mycv::calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);
		cv::goodFeaturesToTrack(opencv_prev_grey, opencv_prev_corner, 200, 0.01, 30);
		cv::calcOpticalFlowPyrLK(opencv_prev_grey, opencv_cur_grey, opencv_prev_corner, opencv_cur_corner, opencv_status, opencv_err);

		// weed out bad matches
		for (size_t i = 0; i < status.size(); i++) {
			if (status[i]) {
				prev_corner2.push_back(prev_corner[i]);
				cur_corner2.push_back(cur_corner[i]);
			}

		}

		for (size_t i = 0; i < opencv_status.size(); i++) {
			if (opencv_status[i]) {
				opencv_prev_corner2.push_back(opencv_prev_corner[i]);
				opencv_cur_corner2.push_back(opencv_cur_corner[i]);
			}
		}

		// translation + rotation only
		Mat T = mycv::estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing
		Mat opencv_T = cv::estimateRigidTransform(opencv_prev_corner2, opencv_cur_corner2, false);

		// in rare cases no transform is found. We'll just use the last known good transform.
		if (T.data == NULL) {
			last_T.copyTo(T);
		}
		if (opencv_T.data == NULL) {
			opencv_last_T.copyTo(opencv_T);
		}

		T.copyTo(last_T);
		opencv_T.copyTo(opencv_last_T);

		// decompose T
		double dx = T.at<double>(0, 2);
		double dy = T.at<double>(1, 2);
		double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));
		double opencv_dx = opencv_T.at<double>(0, 2);
		double opencv_dy = opencv_T.at<double>(1, 2);
		double opencv_da = atan2(opencv_T.at<double>(1, 0), opencv_T.at<double>(0, 0));

		prev_to_cur_transform.push_back(mycv::TransformParam(dx, dy, da));
		opencv_prev_to_cur_transform.push_back(mycv::TransformParam(opencv_dx, opencv_dy, opencv_da));
		//out_transform << k << " " << dx << " " << dy << " " << da << endl;

		cur.copyTo(prev);
		cur.copyTo(opencv_prev);
		cur_grey.copyTo(prev_grey);
		opencv_cur_grey.copyTo(opencv_prev_grey);

		cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;
		k++;
	}

	// Step 2 - Accumulate the transformations to get the image trajectory

	// Accumulated frame to frame transform
	double a = 0;
	double x = 0;
	double y = 0;
	double opencv_a = 0;
	double opencv_x = 0;
	double opencv_y = 0;

	vector <mycv::Trajectory> trajectory; // trajectory at all frames
	vector <mycv::Trajectory> opencv_trajectory;
	for (size_t i = 0; i < prev_to_cur_transform.size(); i++) {
		x += prev_to_cur_transform[i].dx;
		y += prev_to_cur_transform[i].dy;
		a += prev_to_cur_transform[i].da;

		trajectory.push_back(mycv::Trajectory(x, y, a));

		opencv_x += opencv_prev_to_cur_transform[i].dx;
		opencv_y += opencv_prev_to_cur_transform[i].dy;
		opencv_a += opencv_prev_to_cur_transform[i].da;

		opencv_trajectory.push_back(mycv::Trajectory(opencv_x, opencv_y, opencv_a));
		//out_trajectory << (i + 1) << " " << x << " " << y << " " << a << endl;
	}

	// Step 3 - Smooth out the trajectory using an averaging window
	vector <mycv::Trajectory> smoothed_trajectory; // trajectory at all frames
	vector <mycv::Trajectory> opencv_smoothed_trajectory;

	for (size_t i = 0; i < trajectory.size(); i++) {
		double sum_x = 0;
		double sum_y = 0;
		double sum_a = 0;
		int count = 0;
		double opencv_sum_x = 0;
		double opencv_sum_y = 0;
		double opencv_sum_a = 0;
		int opencv_count = 0;

		for (int j = -SMOOTHING_RADIUS; j <= SMOOTHING_RADIUS; j++) {
			if (i + j >= 0 && i + j < trajectory.size()) {
				sum_x += trajectory[i + j].x;
				sum_y += trajectory[i + j].y;
				sum_a += trajectory[i + j].a;

				count++;

				opencv_sum_x += opencv_trajectory[i + j].x;
				opencv_sum_y += opencv_trajectory[i + j].y;
				opencv_sum_a += opencv_trajectory[i + j].a;

				opencv_count++;
			}
		}

		double avg_a = sum_a / count;
		double avg_x = sum_x / count;
		double avg_y = sum_y / count;

		smoothed_trajectory.push_back(mycv::Trajectory(avg_x, avg_y, avg_a));

		double opencv_avg_a = opencv_sum_a / opencv_count;
		double opencv_avg_x = opencv_sum_x / opencv_count;
		double opencv_avg_y = opencv_sum_y / opencv_count;

		opencv_smoothed_trajectory.push_back(mycv::Trajectory(opencv_avg_x, opencv_avg_y, opencv_avg_a));

		//out_smoothed_trajectory << (i + 1) << " " << avg_x << " " << avg_y << " " << avg_a << endl;
	}

	// Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
	vector <mycv::TransformParam> new_prev_to_cur_transform;
	vector <mycv::TransformParam> opencv_new_prev_to_cur_transform;
	// Accumulated frame to frame transform
	a = 0;
	x = 0;
	y = 0;
	opencv_a = 0;
	opencv_x = 0;
	opencv_y = 0;

	for (size_t i = 0; i < prev_to_cur_transform.size(); i++) {
		x += prev_to_cur_transform[i].dx;
		y += prev_to_cur_transform[i].dy;
		a += prev_to_cur_transform[i].da;

		opencv_x += opencv_prev_to_cur_transform[i].dx;
		opencv_y += opencv_prev_to_cur_transform[i].dy;
		opencv_a += opencv_prev_to_cur_transform[i].da;
		// target - current
		double diff_x = smoothed_trajectory[i].x - x;
		double diff_y = smoothed_trajectory[i].y - y;
		double diff_a = smoothed_trajectory[i].a - a;
		double opencv_diff_x = opencv_smoothed_trajectory[i].x - x;
		double opencv_diff_y = opencv_smoothed_trajectory[i].y - y;
		double opencv_diff_a = opencv_smoothed_trajectory[i].a - a;

		double dx = prev_to_cur_transform[i].dx + diff_x;
		double dy = prev_to_cur_transform[i].dy + diff_y;
		double da = prev_to_cur_transform[i].da + diff_a;
		double opencv_dx = opencv_prev_to_cur_transform[i].dx + diff_x;
		double opencv_dy = opencv_prev_to_cur_transform[i].dy + diff_y;
		double opencv_da = opencv_prev_to_cur_transform[i].da + diff_a;

		new_prev_to_cur_transform.push_back(mycv::TransformParam(dx, dy, da));
		opencv_new_prev_to_cur_transform.push_back(mycv::TransformParam(opencv_dx, opencv_dy, opencv_da));
		//out_new_transform << (i + 1) << " " << dx << " " << dy << " " << da << endl;
	}
	// Step 5 - Apply the new transformation to the video
	cap.set(CV_CAP_PROP_POS_FRAMES, 0);
	Mat T(2, 3, CV_64F);
	Mat opencv_T(2, 3, CV_64F);

	int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct
	int opencv_vert_border = HORIZONTAL_BORDER_CROP * opencv_prev.rows / opencv_prev.cols; // get the aspect ratio correct
	vector<Mat> canvases;
	k = 0;

	const int fourcc = -1;
	const int fps = cap.get(CV_CAP_PROP_FPS);
	const string fileName = inputPath.substr(0, inputPath.find_last_of("."));

	VideoWriter outputVideo_compare(fileName + "_compare.avi", fourcc, fps, Size(cur.cols * 3 + 20, cur.rows));
	VideoWriter outputVideo_A(fileName + "_mystab.avi", fourcc, fps, Size(cur.cols, cur.rows));
	VideoWriter outputVideo_B(fileName + "_opencv.avi", fourcc, fps, Size(cur.cols, cur.rows));

	while (k < max_frames - 1) { // don't process the very last frame, no valid transform
		cap >> cur;

		if (cur.data == NULL) {
			break;
		}

		T.at<double>(0, 0) = cos(new_prev_to_cur_transform[k].da);
		T.at<double>(0, 1) = -sin(new_prev_to_cur_transform[k].da);
		T.at<double>(1, 0) = sin(new_prev_to_cur_transform[k].da);
		T.at<double>(1, 1) = cos(new_prev_to_cur_transform[k].da);
		T.at<double>(0, 2) = new_prev_to_cur_transform[k].dx;
		T.at<double>(1, 2) = new_prev_to_cur_transform[k].dy;

		opencv_T.at<double>(0, 0) = cos(opencv_new_prev_to_cur_transform[k].da);
		opencv_T.at<double>(0, 1) = -sin(opencv_new_prev_to_cur_transform[k].da);
		opencv_T.at<double>(1, 0) = sin(opencv_new_prev_to_cur_transform[k].da);
		opencv_T.at<double>(1, 1) = cos(opencv_new_prev_to_cur_transform[k].da);
		opencv_T.at<double>(0, 2) = opencv_new_prev_to_cur_transform[k].dx;
		opencv_T.at<double>(1, 2) = opencv_new_prev_to_cur_transform[k].dy;

		Mat cur2;
		Mat opencv_cur2;
		mycv::warpAffine(cur, cur2, T, cur.size(), 1, BORDER_REFLECT101);
		mycv::warpAffine(cur, opencv_cur2, opencv_T, cur.size(), 1, BORDER_REFLECT101);
		cur2 = cur2(Range(vert_border, cur2.rows - vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols - HORIZONTAL_BORDER_CROP));
		opencv_cur2 = opencv_cur2(Range(opencv_vert_border, opencv_cur2.rows - opencv_vert_border), Range(HORIZONTAL_BORDER_CROP, opencv_cur2.cols - HORIZONTAL_BORDER_CROP));

		// Resize cur2 back to cur size, for better side by side comparison
		mycv::resize(cur2, cur2, cur.size());
		mycv::resize(opencv_cur2, opencv_cur2, cur.size());

		// Now draw the original and stablised side by side for coolness
		Mat canvas = Mat::zeros(cur.rows, cur.cols * 3 + 20, cur.type());

		cur.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
		cur2.copyTo(canvas(Range::all(), Range(cur2.cols + 10, cur2.cols * 2 + 10)));
		opencv_cur2.copyTo(canvas(Range::all(), Range(opencv_cur2.cols * 2 + 20, opencv_cur2.cols * 3 + 20)));

		// If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
		if (canvas.cols > 1920) {
			mycv::resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));
		}

		canvases.push_back(canvas.clone());
		outputVideo_compare << canvas.clone();
		outputVideo_A << cur2;
		outputVideo_B << opencv_cur2;

		k++;
	}
	outputVideo_compare.release();
	outputVideo_A.release();
	outputVideo_B.release();
	return 0;
}
