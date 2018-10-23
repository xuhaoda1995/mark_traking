// TrafficSign.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <string>
#include "TrafficSingDetector.h"
#include "HogSvmClassifier.h"
#include <iostream>
#include <time.h>
#include <fstream>

//////////////////////////////////////////////////////////////////////////
//要载入的分类器名称，TS_CLASSIFIER_NUM定义见TS_define.h
string trafficSignName2[TS_CLASSIFIER_NUM] = { "注意行人", "禁止停车", "施工" };

//////////////////////////////////////////////////////////////////////////
//路径

struct sign
{
	Rect rect;
	uchar type;
};


/*************************************************************************/
/*用于训练三角警示牌数据
/*************************************************************************/
int main1()
{
	TrafficSingDetector detector;
	HogSvmClassifier classifiers;

	std::string strPath = "./images/originData.txt";

	unsigned int ErrNum = 0;
	int a = classifiers.train(strPath, &ErrNum);

	strPath = "./images/warnTriangle.xml";
	classifiers.svm_save(strPath);


	return 0;
	return 0;
}