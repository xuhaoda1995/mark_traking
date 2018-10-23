#ifndef TRAFFIC_SIGN_DETECTOR_H_
#define TRAFFIC_SIGN_DETECTOR_H_

#include <opencv2/opencv.hpp>

class TrafficSingDetector
{
public:
	TrafficSingDetector();
	~TrafficSingDetector();
private:
	cv::Size img_size;

public:
	int saturation_detect(cv::Mat *img, std::vector<cv::Rect> *signs, int thresh_value);

};

#endif