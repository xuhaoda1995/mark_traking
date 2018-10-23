// TrafficSign.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <string>
#include "TrafficSingDetector.h"
#include "HogSvmClassifier.h"
#include <iostream>
#include <time.h>
#include <fstream>

//////////////////////////////////////////////////////////////////////////
//Ҫ����ķ��������ƣ�TS_CLASSIFIER_NUM�����TS_define.h
string trafficSignName2[TS_CLASSIFIER_NUM] = { "ע������", "��ֹͣ��", "ʩ��" };

//////////////////////////////////////////////////////////////////////////
//·��

struct sign
{
	Rect rect;
	uchar type;
};


/*************************************************************************/
/*����ѵ�����Ǿ�ʾ������
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