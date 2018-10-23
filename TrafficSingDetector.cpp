#include "stdafx.h"
#include "TrafficSingDetector.h"
#include <iostream>

using namespace std;
using namespace cv;

TrafficSingDetector::TrafficSingDetector()
{
	img_size.width = 0;
	img_size.height = 0;
}


TrafficSingDetector::~TrafficSingDetector()
{
}


int TrafficSingDetector::saturation_detect(Mat *img, vector<Rect> *signs, int  thresh_value)
{
	if (img_size.width == 0 || img_size.height == 0)
	{
		img_size.width = img->cols;
		img_size.height = img->rows;
	}
	else if (img_size.width != img->cols || img_size.height != img->rows)
	{
		img_size.width = img->cols;
		img_size.height = img->rows;
	}

	if (!signs->empty())
		signs->clear();

	Mat hsv_img = img->clone();
	Mat hsv[3];
	cvtColor(hsv_img, hsv_img, COLOR_RGB2HSV);
	split(hsv_img, hsv);

	// imshow("sh", hsv[0]);
	// waitKey(0);

	Mat s_th;
	threshold(hsv[0], s_th, thresh_value, 255, THRESH_BINARY);

	// imshow("sh", s_th);
	// waitKey(0);

	int kernel_size = (int)(((img_size.width + img_size.height) / 320));
  	if (kernel_size % 2 != 1)
		kernel_size++;
	medianBlur(s_th, s_th, kernel_size);

	// imshow("sh", s_th);
	// waitKey(0);

	vector<vector<Point> > contours;
	findContours(s_th,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

	for (auto contour : contours)
	{
		Rect r = boundingRect(contour);
		float asp = (float)(r.width) / (float)(r.height);
		Mat v_roi = hsv[2](r);
		Scalar v_avg = mean(v_roi);
		int extend_pix = 0;
		/*if (r.height > 10 && r.width > 10
			&& v_avg[0] > 50
			)*/
		if (r.height > 10 && r.width > 10
			&& v_avg[0] > 5 && r.height < 20 && r.width < 20
			)
		{
			extend_pix = (r.width + r.height) / 40;
			(r.x - extend_pix) < 0 ? r.x = 0 : r.x -= extend_pix;
			(r.y - extend_pix) < 0 ? r.y = 0 : r.y -= extend_pix;
			(r.width + r.x + extend_pix * 2 > img_size.width) ? r.width = img_size.width - r.x : r.width += extend_pix * 2;
			(r.height + r.y + extend_pix * 2 > img_size.height) ? r.height = img_size.height - r.y : r.height += extend_pix * 2;
			signs->push_back(r);
			//rectangle(*img, r, Scalar(0, 255, 0), 2);
		}
	}

	return 0;
}