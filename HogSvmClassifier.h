#ifndef HOG_SVM_CLASSIFIER_H_
#define HOG_SVM_CLASSIFIER_H_

#include <opencv2/opencv.hpp>

#include "TS_define.h"
#include <string>
using namespace std;
using namespace cv;
class HogSvmClassifier
{

public:
	HogSvmClassifier();
	~HogSvmClassifier();

//////////////////////////////////////////////////////////////////////////
private:
	unsigned int sample_num;  
	unsigned int imgHeight;  
	unsigned int imgWidth;   
	Mat feature_mat;        
	Mat label_mat;          

private:
	HOGDescriptor *hog;   
	size_t hog_size;    
	Ptr<ml::SVM> svm;

private:
	int featureDataPreparedFlag;    
	int labelDataPreparedFlag;     
	int svmTraindFlag;

public:
	int train(string txt_path, unsigned int *ErrNum);
	int train(string img_txt_paths, string label_txt_path, unsigned int *ErrNum);
	int predict(Mat *img, float *class_type);
	int svm_save(string path);
	int svm_load(string path);
	int print_info();
	int change_info();

private:
	int load_labels(string path);
	int hog_featere(Mat *img, int j);
	int hog_featere_pred(Mat *img, Mat *pred_mat);
	int hog_featere_with_path(string txt_path, unsigned int *ErrNum, int LoadType = HOG_LOAD_TYPE_SEP);
	int svm_train();
};

#endif