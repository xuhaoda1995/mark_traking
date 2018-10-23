#include "stdafx.h"
#include "HogSvmClassifier.h"
#include <assert.h>
#include <vector>
#include <fstream>

#include <iostream>

HogSvmClassifier::HogSvmClassifier()
{
	sample_num = 0;
	imgHeight = 32;
	imgWidth = 32;
	featureDataPreparedFlag = 0;
	labelDataPreparedFlag = 0;
	svmTraindFlag = 0;
	hog = new HOGDescriptor(Size(imgWidth, imgHeight), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	hog_size = hog->getDescriptorSize();
	svm=ml::SVM::create();

	svm->setType(cv::ml::SVM::Types::C_SVC);
	svm->setKernel(cv::ml::SVM::KernelTypes::RBF);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

}



HogSvmClassifier::~HogSvmClassifier()
{
	delete hog;
	//cvReleaseMat(&feature_mat);
	//cvReleaseMat(&label_mat);
	//cvReleaseMat(&pred_mat);
	//if (!feature_mat->empty())
		//cvSetZero(feature_mat);
	//feature_mat->release();

	//if (!label_mat->empty())
		//cvSetZero(feature_mat);
	//label_mat->release();
}


int HogSvmClassifier::load_labels(string path)
{
	ifstream txtFile(path.c_str());
	unsigned int num = 0;
	try
	{
		txtFile >> num;
	}
	catch (const std::exception&)
	{
		txtFile.close();
		return -1;
	}
	if (featureDataPreparedFlag)
		if (sample_num != num)
		{
			txtFile.close();
			return -2;
		}		
		
	if (!labelDataPreparedFlag)
	{
		if (!featureDataPreparedFlag)
			sample_num = num;
		label_mat = Mat::zeros(sample_num, 1, CV_32FC1);
	}

	cout << "Read  Labels..." << endl;
	for (unsigned int i = 0; i < num; ++i)
	{
		cout << i << endl;
		//txtFile >> label_mat->data[i];
		float k;
		txtFile >> k;
		// cvmSet(label_mat, i, 0, k);
		label_mat.at<float>(i,0)=k;
	}
		
	//////////////////////////////////////////////////////////////////////////
	labelDataPreparedFlag = 1;
	txtFile.close();

	return 0;
}

int HogSvmClassifier::hog_featere(Mat *img, int j)
{
	if (img->cols != imgWidth || img->rows != imgHeight)
		resize(*img, *img, Size(imgWidth, imgHeight), INTER_CUBIC);

	vector<float> descriptors((int)hog_size);

	hog->compute(*img, descriptors, Size(1, 1), Size(0, 0));

	// float *data = (float*)feature_mat.data;
	// for (vector<float>::size_type i = 0; i < descriptors.size(); ++i)
	// 	data[i] = descriptors[i];

	return 0;
}



/**************************************************************************************/
/* ���ܣ����ڼ���HOG������ȡ  */
/* ����ֵ��
		  0��ִ�гɹ�  */
/**************************************************************************************/
int HogSvmClassifier::hog_featere_pred(Mat *img, Mat *pred_mat)
{
	if (img->cols != imgWidth || img->rows != imgHeight)
		resize(*img, *img, Size(imgWidth, imgHeight), INTER_CUBIC);

	vector<float> descriptors((int)hog_size);

	hog->compute(*img, descriptors, Size(1, 1), Size(0, 0));

	float *data = (float*)pred_mat->data;
	for (vector<float>::size_type i = 0; i < descriptors.size(); ++i)
		data[i] = descriptors[i];

	return 0;
}



/*************************************************************************************/
/* ���ܣ�����txt�����·����������ȡͼƬHOG����  */
/* ����ֵ��
		  0��ִ�гɹ�
		  -1: ��ȡ����ʧ��
		  -2: ͼ��������ǩ������ƥ��
		  -3: ͼ���ȡʧ��
          -4������δ֪�쳣
LoadType����ȡ��ʽ�����TS_define.h
ErrNum������ʧ�ܵ�·�����  */
/*************************************************************************************/
int HogSvmClassifier::hog_featere_with_path(string txt_path, unsigned int *ErrNum, int LoadType)
{
	*ErrNum = 0;
	unsigned int num = 0;
	string img_path; 
	//////////////////////////////////////////////////////////////////////////
	//��ȡͼƬ���������Ѷ�ȡ��ǩ���ж�ͼƬ�����Ƿ����ǩƥ��
	ifstream txtFile(txt_path.c_str());
	try
	{
		txtFile >> num;
	}
	catch (const std::exception& )
	{
		txtFile.close();
		return -1;
	}
	if (labelDataPreparedFlag)
		if (label_mat.rows != num)
		{
			txtFile.close();
			return -2;
		}
	//////////////////////////////////////////////////////////////////////////
	//Ϊ��������/��ǩ����ռ䲢ȷ����������
	if (!featureDataPreparedFlag)
	{
		if (!labelDataPreparedFlag)
			sample_num = num;
		feature_mat = Mat::zeros(sample_num, (int)hog_size, CV_32FC1);
	}
	if (LoadType == HOG_LOAD_TYPE_TOG)
		if (!labelDataPreparedFlag)
		{
			label_mat = Mat::zeros(sample_num, 1, CV_32FC1);
		}
	//////////////////////////////////////////////////////////////////////////
	//��ȡͼƬ������HOGֵ
	float *data = NULL;
	cout << "Start Compute HOG Features ..." << endl;
	try
	{
		for (unsigned int i = 0; i < num; ++i)
		{
			cout << i << endl;
			*ErrNum = i + 1;
			txtFile >> img_path;
			if (LoadType == HOG_LOAD_TYPE_TOG)
			{
				float k;
				txtFile >> k;
				label_mat.at<float>(i, 0)= k;
				//txtFile >> label_mat->data[i];
			}
			Mat img = imread(img_path.c_str(), 1);
			if (img.empty())
			{
				txtFile.close();
				return -3;
			}
			hog_featere(&img, i);
		}
	}
	catch (const std::exception&)
	{
		txtFile.close();
		return -4;
	}
	//////////////////////////////////////////////////////////////////////////
	//���ñ�־λ���ر��ļ���
	featureDataPreparedFlag = 1;
	if (LoadType == HOG_LOAD_TYPE_TOG)
		labelDataPreparedFlag = 1;
	txtFile.close();

	return 0;
}



/**************************************************************************************/
/* ���ܣ�����SVMѵ��  */
/* ����ֵ��
		0��ִ�гɹ�
		-1: �Ѵ���SVMģ��
		-2: �����Ż����粻ƥ�� 
		-3: ������ѵ��ʧ��  */
/**************************************************************************************/
int HogSvmClassifier::svm_train()
{
	if (svmTraindFlag)
		return -1;
	//////////////////////////////////////////////////////////////////////////
	//���ò����Ż�����
	// CvParamGrid CvParamGrid_C(pow(2.0, -5), pow(2.0, 15), pow(2.0, 2));
	// CvParamGrid CvParamGrid_gamma(pow(2.0, -15), pow(2.0, 3), pow(2.0, 2));
	// if (!CvParamGrid_C.check() || !CvParamGrid_gamma.check())
	// 	return - 2;
	//////////////////////////////////////////////////////////////////////////
	//��ʼѵ��
	cout << "Start Train SVM Module ..." << endl;
	try
	{
		// svm.train_auto(feature_mat, label_mat, Mat(), Mat(), param, 10, CvParamGrid_C, CvParamGrid_gamma,
		// 	CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);
	}
	catch (const std::exception&)
	{
		return -3;
	}
	//////////////////////////////////////////////////////////////////////////
	//����ѵ����ɱ�־λ
	cout << "SVM Module Training Completed ! " << endl;
	svmTraindFlag = 1;

	return 0;
}



/**************************************************************************************/
/* ���ܣ�����������HOG������ȡ + SVMѵ��  */
/* ͼƬ·���ͱ�ǩ���ֱ�д������txt�ļ��У��ļ���ͷ��һ�о�Ϊ��������  */
/* ����ֵ��
		  0��ִ�гɹ�
		  -1: ��ȡ����ʧ��
		  -2: ͼ��������ǩ������ƥ��
		  -3: ͼ���ȡʧ��
		  -4������δ֪�쳣
		  -5���Ѵ���SVMģ��
		  -6: SVM�����Ż����粻ƥ��
		  -7: SVM������ѵ��ʧ��  */
/**************************************************************************************/
int HogSvmClassifier::train(string img_txt_paths, string label_txt_path, unsigned int *ErrNum)
{
	int res = 0;
	if (res = hog_featere_with_path(img_txt_paths, ErrNum, HOG_LOAD_TYPE_SEP) != 0)
		return res;
	if (res = load_labels(label_txt_path) != 0)
		return res;
	if (labelDataPreparedFlag && featureDataPreparedFlag)
		if (res = svm_train() != 0)
			return res - 4;

	return 0;
}



/**************************************************************************************/
/* ���ܣ�����������HOG������ȡ + SVMѵ��  */
/* ͼƬ·���ͱ�ǩд��һ��txt�ļ��У��ļ���ͷ��һ��Ϊ����������ͼƬ·�����ͼƬ��ǩ  */
/* ����ֵ��
		  0��ִ�гɹ�
		  -1: ��ȡ����ʧ��
		  -2: ͼ��������ǩ������ƥ��
		  -3: ͼ���ȡʧ��
		  -4������δ֪�쳣
		  -5���Ѵ���SVMģ��
		  -6: SVM�����Ż����粻ƥ��
		  -7: SVM������ѵ��ʧ��  */
/**************************************************************************************/
int HogSvmClassifier::train(string txt_path, unsigned int *ErrNum)
{
	int res = 0;
	if (res = hog_featere_with_path(txt_path, ErrNum, HOG_LOAD_TYPE_TOG) != 0)
		return res;
	if (featureDataPreparedFlag&&labelDataPreparedFlag)
		if (res = svm_train() != 0)
			return res - 5;

	return 0;
}


int HogSvmClassifier::predict(Mat *img, float *class_type)
{
	int res = 0;
	if (svmTraindFlag)
	{
		Mat *pred_mat = new Mat(1, (int)hog_size, CV_32FC1);
		res = hog_featere_pred(img, pred_mat);
		*class_type = svm->predict(*pred_mat);
		delete pred_mat;
	}
	else
		return -1;

	return 0;
}



/**************************************************************************************/
/* ���ܣ�����Ŀǰʹ�õ�SVMģ�ͣ�XML�ļ���  */
/* ����ֵ��
		  0��ִ�гɹ�
		  -1����SVMģ�ͣ�δѵ����δ���룩 */
/**************************************************************************************/
int HogSvmClassifier::svm_save(string path)
{
	if (svmTraindFlag)
		svm->save(path.c_str());
	else
		return -1;

	return 0;
}



/**************************************************************************************/
/* ���ܣ�����SVMģ�͵��ࣨXML�ļ���  */
/* ����ֵ��
		  0��ִ�гɹ�  
		  -1: �Ѵ���SVMģ��  */
/**************************************************************************************/
int HogSvmClassifier::svm_load(string path)
{
	if (svmTraindFlag)
		return -1;
	svm=ml::SVM::load(path.c_str());
	svmTraindFlag = 1;

	return 0;
}



int HogSvmClassifier::print_info()
{
	return 0;
}



int HogSvmClassifier::change_info()
{
	return 0;
}