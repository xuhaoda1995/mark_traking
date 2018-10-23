#include "stdafx.h"
#include <string>
#include "TrafficSingDetector.h"
#include "HogSvmClassifier.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <opencv2/tracking/tracker.hpp>

struct sign
{
	Rect rect;
	uchar type;
};

// void subBackground()
// {
// 		Mat candidate;
// 		subtract(img, backgraound, candidate);
// 		imshow("res", candidate);
// 		waitKey(0);

// 		cvtColor(candidate,candidate,COLOR_BGR2GRAY);
// 		imshow("res", candidate);
// 		waitKey(0);

// 		threshold(candidate, candidate, 50, 255, THRESH_OTSU);
// 		imshow("res", candidate);
// 		waitKey(0);

// 		// int kernel_size = (int)(((candidate.cols + candidate.rows) / 320));
// 		// if (kernel_size % 2 != 1)
// 		// 	kernel_size++;

// 		// medianBlur(candidate, candidate, kernel_size);
// 		// imshow("res", candidate);
// 		// waitKey(0);

// 		vector<vector<Point>> contours;
// 		findContours(candidate, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
// 		imshow("res", candidate);
// 		waitKey(0);
// }

int main()
{
	TrafficSingDetector detector;
	HogSvmClassifier classifier;

	string load_path = "/home/howstar/haigong/mark_tracking/triangle.xml";
	int err = classifier.svm_load(load_path);
	cout << load_path << ": " << err << endl;

	TrackerMedianFlow::Params param;
	Ptr<Tracker> tracker = TrackerMedianFlow::create(param);

	cv::VideoCapture capture("/home/howstar/haigong/video/1.avi");

	long totalFrameNumber = capture.get(CAP_PROP_FRAME_COUNT); //获取视频的总帧数
	cout << "整个视频共" << totalFrameNumber << "帧" << endl;
	//设置开始帧
	long frameToStart = 0;
	capture.set(CAP_PROP_POS_FRAMES, frameToStart);
	cout << "从第" << frameToStart << "帧开始读" << endl;
	if (!capture.isOpened())
		return -1;

	int nFrame = 0;
	int nTracker = 0;
	Rect2d roi;
	Mat img;

	// Mat backgraound;
	// capture >> backgraound;
	// resize(backgraound, backgraound, Size(960, 640), CV_INTER_CUBIC);

	// namedWindow("res", WINDOW_NORMAL);
	for (;;)
	{
		capture >> img;
		// img=imread("/home/howstar/haigong/mark_tracking/test2.jpg");
		++nFrame;
		//Mat img = imread("test2.jpg", 1);
		if (img.empty())
			continue;

		vector<sign> traffic_signs;
		double runTime = (double)cv::getTickCount();
		resize(img, img, Size(960, 640), CV_INTER_CUBIC);

		if (nTracker == 0)
		{
			//////////////////////////////////////////////////////////////////////////
			vector<Rect> candidates;
			detector.saturation_detect(&img, &candidates, 90);

			float result = 0.0;
			for (auto candidate : candidates)
			{
				//Rect r = boundingRect(contour);
				Mat dete_img = img(candidate);
				// imshow("res", dete_img);
				// waitKey(0);

				result = 0.0;
				classifier.predict(&dete_img, &result);
				if (result == 1.0)
				{
					sign s;
					s.rect = candidate;
					s.type = (uchar)0;
					traffic_signs.push_back(s);
				}
			}
		}
		//////////////////////////////////////////////////////////////////////////
		//for (unsigned int i = 0; i < traffic_signs.size(); ++i)
		if (traffic_signs.size() > 0 || nTracker > 0)
		{
			nTracker++;
			if (nTracker == 1)
			{
				roi = traffic_signs[0].rect;
				tracker->init(img, roi);
				// continue;
			}
			else
			{
				tracker->update(img, roi);
			}
			rectangle(img, roi, Scalar(0, 255, 0), 2);
			Point oriPoint;
			oriPoint.x = roi.x;
			oriPoint.y = roi.y + 10;
			putText(img, "Triangle", oriPoint, FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 255, 0));

			cout << nFrame << "\t"
				 << "Triangle"
				 << ": " << roi.x << " "
				 << roi.y << " " << roi.width
				 << " " << roi.height << endl;
		}

		runTime = ((double)cv::getTickCount() - runTime) / (double)cv::getTickFrequency() * 1000;
		cout << "the time used is " << runTime << "ms" << endl;

		imshow("result", img);
		if (waitKey(1) == 27)
			break;
	}

	cv::destroyAllWindows();

	return 0;
}