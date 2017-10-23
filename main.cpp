#include<opencv/cv.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/gpu/gpu.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
using namespace cv;
using namespace std;

struct T
{
	double width;
	double height;
	int r;
};

vector<T>Params;

void OnDrawDotline(CvPoint s, CvPoint d, IplImage *img)
{
	CvPoint pa, pb;

	double k = (s.y - d.y) / (s.x - d.x + 0.000001);

	double h = img->height;
	double w = img->width;

	pa.x = w;
	pa.y = s.y + k*(w - s.x);
	pb.y = d.y - k*d.x;
	pb.x = 0;

	cvLine(img, pb, pa, CV_RGB(255, 0, 0), 3, CV_AA, 0);

}
Mat HE(Mat origin)   //将直方图归一化后累加，再根据映射将其均衡化
{
	//double color[255];
	double *color = new double[255];
	Mat output = origin;
	for (int i = 0; i < 255; i++)
		color[i] = 0;
	for (int i = 0; i < origin.rows; i++)
	for (int j = 0; j < origin.cols; j++)
		output.data[i*origin.cols + j] = origin.data[i*origin.cols + j];
	for (int i = 0; i < origin.rows; i++)
	for (int j = 0; j < origin.cols; j++)
		color[origin.data[i*origin.cols + j]]++;
	for (int i = 0; i < 255; i++)
	{
		color[i] = (double)color[i] / (origin.rows*origin.cols);
	}
	for (int i = 1; i < 255; i++)
	{
		color[i] = color[i] + color[i - 1];
	}
	for (int i = 0; i < origin.rows; i++)
	for (int j = 0; j < origin.cols; j++)
		output.data[i*origin.cols + j] = color[origin.data[i*origin.cols + j]] * 255;
	return output;
}


#define TRAIN//是否训练还是直接载入训练好的模型

class MySVM : public CvSVM
{
public:
	double * get_alpha_data()
	{
		return this->decision_func->alpha;
	}
	double  get_rho_data()
	{
		return this->decision_func->rho;
	}
};
int heightMax, widthMax, heightMin, widthMin;
int border = 30;
int border1 = 100;
bool isRorB(int b, int g, int r)
{
	return true;
	if ((r > 150) && (g < 140) && (b < 120))
		return true;
	if ((b > 160) && (r < 100) && (g < 150))
		return true;
	if ((b > 100) && (r < 50) && (g < 50))
		return true;
	return false;
}
void main(int argc, char ** argv)
{
#if 1

	for (int i = 1; i < 48; i++)
	{
		char buffer[3];
		_itoa(i, buffer, 10);
		string tmp = buffer;
		//ss >> tmp;
		string filename = "pics\\stage1\\" + tmp + ".jpg";
		string donename = "pics\\base\\" + tmp + ".jpg";
		Mat fileMat = imread(filename,-1);

		int r1 = fileMat.rows / 800;
		int r2 = fileMat.cols / 800;
		if ((r1 > 1) || (r2 > 1))
		{
			int r = r1 > r2 ? r1 : r2;
			Size dsize = Size(fileMat.cols / r, fileMat.rows / r);
			//tmpMat = imread(desName,-1);
			Mat trainImg = Mat(dsize, CV_32S);
			resize(fileMat, trainImg, dsize);
			imwrite(donename, trainImg);
			//imshow("test", trainImg);
		}
		else
		{
			imwrite(donename, fileMat);
		}

	}
#endif
#if 1

	for (int i = 1; i < 48; i++)
	{


		char buffer[3];
		_itoa(i, buffer, 10);
		string tmp = buffer;
		//ss >> tmp;
		string filename = "pics\\base\\" + tmp + ".jpg";
		if (tmp.size() == 1)
			tmp = "00" + tmp;
		if (tmp.size() == 2)
			tmp = "0" + tmp;
		string donename = "pics\\test\\" + tmp + ".png";
		IplImage *imgSource = cvLoadImage(filename.c_str(), -1);
		if (imgSource == NULL)
		{
			cout << "load error";
			return;
		}
		int height = imgSource->height;
		int width = imgSource->width;
		//cout << width;
		int widthStep = imgSource->widthStep;
		heightMax = widthMax = 0;
		heightMin = height;
		widthMin = width;
		if (imgSource != 0)//imgSourceΪIplImage*
		{
			for (int i = 0; i < height; ++i)

			{
				uchar * pucPixel = (uchar*)imgSource->imageData + i*widthStep;

				for (int j = 0; j < width; ++j)
				{

					if (isRorB(pucPixel[3 * j], pucPixel[3 * j + 1], pucPixel[3 * j + 2]))
					{
						if (i>heightMax)
							heightMax = i;
						if (i < heightMin)
							heightMin = i;
						if (j>widthMax)
							widthMax = j;
						if (j < widthMin)
							widthMin = j;
					}

				}

			}

		}


		widthMax = widthMax + border;
		if (widthMax > width)
			widthMax = width;
		widthMin = widthMin - border;
		if (widthMin < 0)
			widthMin = 0;

		heightMax = heightMax + border1;
		if (heightMax > height)
			heightMax = height;
		heightMin = heightMin - border;
		if (heightMin < 0)
			heightMin = 0;

		int newWidth = widthMax - widthMin;
		int newHeight = heightMax - heightMin;
		if ((newWidth < 0) || (newWidth < 0))
		{
			T tmp;
			tmp.r = 1;
			tmp.width = 0;
			tmp.height = 0;
			Params.push_back(tmp);
			cvSaveImage(donename.c_str(), imgSource);
			Mat srcImage = imread(donename.c_str(), 0);
			Mat newImage = HE(srcImage);
			imwrite(donename.c_str(), newImage);
			cout << "ERROR " << i << endl;
			continue;
		}

		CvSize size = cvSize(newWidth, newHeight);
		cvSetImageROI(imgSource, cvRect(widthMin, heightMin, newWidth, newHeight));

		IplImage *img_mean = cvCreateImage(cvSize(newWidth, newHeight), 8, 3);
		cvCopy(imgSource, img_mean);
		cvResetImageROI(imgSource);
		cvSaveImage(donename.c_str(), img_mean);
		Mat srcImage = imread(donename.c_str(), 0);
		Mat newImage = HE(srcImage);
		{
			T tmp;
			tmp.r = 1;
			tmp.width = widthMin;
			tmp.height = heightMin;
			Params.push_back(tmp);
		imwrite(donename.c_str(), newImage);
		}



	}




#endif

#if 1

	MySVM SVM;
	int descriptorDim;
	string buffer;
	string trainImg;
	vector<string> posSamples;
	vector<string> negSamples;
	vector<string> testSamples;
	int posSampleNum;
	int negSampleNum;
	int testSampleNum;
	const int size = 32;
	//string basePath = "";//相对路径之前加上基地址，如果训练样本中是相对地址，则都加上基地址
	string testbasePath = "pics/test/";
	string posbasePath = "pics/pos16/";
	string negbasePath = "pics/neg/";
	double rho;



#ifdef TRAIN
	for (int i = 1; i < 204; i++)
	{
		char buffer[3];
		_itoa(i, buffer, 10);
		string tmp = buffer;
		if (tmp.size() == 1)
			tmp = "00" + tmp;
		if (tmp.size() == 2)
			tmp = "0" + tmp;
		string filename = posbasePath +tmp + ".png";
		posSamples.push_back(filename);
	}

	posSampleNum = posSamples.size();
	for (int i = 1; i < 114; i++)
	{
		char buffer[3];
		_itoa(i, buffer, 10);
		string tmp = buffer;
		
		string filename = negbasePath+ "0 (" + tmp + ").png";
		negSamples.push_back(filename);
	}
	negSampleNum = negSamples.size();

	Mat sampleFeatureMat;//样本特征向量矩阵
	Mat sampleLabelMat;//样本标签

	HOGDescriptor * hog = new HOGDescriptor(cvSize(size, size), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	vector<float> descriptor;

	for (int i = 0; i < posSampleNum; i++)// 处理正样本
	{
		Mat inputImg = imread(posSamples[i], IMREAD_GRAYSCALE);
		std::cout << "processing " << i << "/" << posSampleNum << " " << posSamples[i] << endl;
		Size dsize = Size(size, size);
		Mat trainImg = Mat(dsize, CV_32S);
		resize(inputImg, trainImg, dsize);
		descriptor.clear();
		hog->compute(trainImg, descriptor, Size(1, 1), Size(0, 0));
		//hog->compute(trainImg, descriptor, Size(8, 8));
		descriptorDim = descriptor.size();

		if (i == 0)//首次特殊处理根据检测到的维数确定特征矩阵的尺寸
		{
			sampleFeatureMat = Mat::zeros(posSampleNum + negSampleNum, descriptorDim, CV_32FC1);
			sampleLabelMat = Mat::zeros(posSampleNum + negSampleNum, 1, CV_32FC1);
		}

		for (int j = 0; j < descriptorDim; j++)//复制特征向量
		{
			sampleFeatureMat.at<float>(i, j) = descriptor[j];
		}

		sampleLabelMat.at<float>(i, 0) = 1;
		//sampleLabelMat.at<float>(i, 0) = 1;
	}
	cout << "extract posSampleFeature done" << endl;

	for (int i = 0; i < negSampleNum; i++)//处理负样本
	{
		Mat inputImg = imread(negSamples[i]);
		cout << "processing " << i << "/" << negSampleNum << " " << negSamples[i] << endl;
		Size dsize = Size(size, size);
		Mat trainImg = Mat(dsize, CV_32S);
		resize(inputImg, trainImg, dsize);
		descriptor.clear();

		hog->compute(trainImg, descriptor, Size(1, 1), Size(0, 0));
		//hog->compute(trainImg, descriptor, Size(8, 8));
		descriptorDim = descriptor.size();

		for (int j = 0; j < descriptorDim; j++)
		{
			sampleFeatureMat.at<float>(posSampleNum + i, j) = descriptor[j];
		}

		sampleLabelMat.at<float>(i, 0) = 1;
		//sampleLabelMat.at<float>(posSampleNum + i, 0) = -1;
	}
	cout << "extract negSampleFeature done" << endl;
	ofstream foutFeature("SampleFeatureMat.txt");//保存特征向量文件
	for (int i = 0; i < posSampleNum + negSampleNum; i++)
	{
		for (int j = 0; j < descriptorDim; j++)
		{
			foutFeature << sampleFeatureMat.at<float>(i, j) << " ";
		}
		foutFeature << "\n";
	}
	foutFeature.close();
	cout << "output posSample and negSample Feature done" << endl;

	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	CvSVMParams params = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);
	cout << "SVM Training Start..." << endl;
	SVM.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), params);

	SVM.save("SVM_Model.xml");
	cout << "SVM Training Complete" << endl;
#endif

#ifndef TRAIN
	SVM.load("SVM_Model.xml");//加载模型文件
#endif
	descriptorDim = SVM.get_var_count();
	int supportVectorNum = SVM.get_support_vector_count();
	cout << "support vector num: " << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);
	Mat supportVectorMat = Mat::zeros(supportVectorNum, descriptorDim, CV_32FC1);
	Mat resultMat = Mat::zeros(1, descriptorDim, CV_32FC1);

	for (int i = 0; i < supportVectorNum; i++)//复制支持向量矩阵
	{
		const float * pSupportVectorData = SVM.get_support_vector(i);
		for (int j = 0; j < descriptorDim; j++)
		{
			supportVectorMat.at<float>(i, j) = pSupportVectorData[j];
		}
	}

	double *pAlphaData = SVM.get_alpha_data();
	for (int i = 0; i < supportVectorNum; i++)//复制函数中的alpha 记住决策公式Y= wx+b
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	resultMat = -1 * alphaMat * supportVectorMat; //alphaMat就是权重向量

	//cout<<resultMat;

	cout << "描述子维数 " << descriptorDim << endl;
	vector<float> myDetector;
	for (int i = 0; i < descriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}

	rho = SVM.get_rho_data();
	myDetector.push_back(rho);
	cout << "检测子维数 " << myDetector.size() << endl;

	HOGDescriptor myHOG(cv::Size(size, size), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9, 1, -1.0, 0, 0.2, true);
	myHOG.setSVMDetector(myDetector);//设置检测子
	//保存检测子
	int minusNum = 0;
	int posNum = 0;
	ofstream foutDetector("HogDetectorForCarFace.txt");
	for (int i = 0; i < myDetector.size(); i++)
	{
		foutDetector << myDetector[i] << " ";
	}
	foutDetector.close();
	//test part

	ifstream fInTest("train/testSample.txt");
	while (fInTest)
	{
		if (getline(fInTest, buffer))
		{
			testSamples.push_back(testbasePath + buffer);
		}
	}
	testSampleNum = testSamples.size();
	fInTest.close();



	
	int count = 1;

	for (int i = 0; i < testSamples.size(); i++)
	{
		int hardExampleCount = 1;
		Mat testImg = imread(testSamples[i]);
		Mat testImgNorm = testImg.clone();//复制原图 
		vector<Rect> found;
		cout << "MultiScale detect " << endl;
		myHOG.detectMultiScale(testImgNorm, found, 0, Size(8, 8), Size(size, size), 1.03, 2, false);
		cout << "Detected Rect Num" << found.size() << endl;
		for (int j = 0; j < found.size(); j++)
		{
			//将超出了图像边界的矩形框规范在图像边界内部  
			Rect r = found[j];
			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > testImgNorm.cols)
				r.width = testImgNorm.cols - r.x;
			if (r.y + r.height > testImgNorm.rows)
				r.height = testImgNorm.rows - r.y;
			//保存Hard Example  
			char saveName[256];
			if ((r.width > 0) && (r.height > 0))
			{
			
			Mat hardExampleImg = testImgNorm(r);//从原图上截取矩形框大小的图片  
			resize(hardExampleImg, hardExampleImg, Size(size, size));//将剪裁出来的图片缩放为64*64大小  
			sprintf(saveName, "hardExample/%d(%03d).png",i, hardExampleCount++);
			imwrite(saveName, hardExampleImg);//保存文件 
			
		}
	}

		char buffer[3];
		_itoa(i+1, buffer, 10);
		string tmp = buffer;
		string filename = "pics\\base\\" + tmp + ".jpg";








		IplImage* src = cvLoadImage(filename.c_str(), 0);
		IplImage* dst;
		IplImage* color_dst;
		CvMemStorage* storage = cvCreateMemStorage(0);
		CvSeq* lines = 0;

	
		dst = cvCreateImage(cvGetSize(src), 8, 1);
		color_dst = cvCreateImage(cvGetSize(src), 8, 3);
		cvCanny(src, dst, 50, 200, 3);
		cvCvtColor(dst, color_dst, CV_GRAY2BGR);

			int maxlength = 0;
		CvPoint* maxline = NULL;
		lines = cvHoughLines2(dst, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI / 180, 107, 30, 3);
		for (int n = 0; n < lines->total; n++)
		{
			CvPoint* line = (CvPoint*)cvGetSeqElem(lines, n);
			double k = (line[1].y - line[0].y) / (line[1].x - line[0].x + 0.000001);
			if (k >= 0.3 || k <= -0.3)
				continue;
			if (sqrt((line[0].x - line[1].x)*(line[0].x - line[1].x) + (line[0].y - line[1].y)*(line[0].y - line[1].y)) >= maxlength) {
				maxlength = sqrt((line[0].x - line[1].x)*(line[0].x - line[1].x) + (line[0].y - line[1].y)*(line[0].y - line[1].y));
				maxline = line;
			}
		}
		if (maxline != NULL)
		{
			IplImage* DES = cvLoadImage(filename.c_str(), -1);
			OnDrawDotline(maxline[0], maxline[1], DES);
			cvSaveImage(filename.c_str(),DES);
			cout << "find line\n";
		}
		else
		{
			cout << "can`t find line\n";
		}

		T tmpT = Params[i];
		Mat tmpMat = imread(filename, -1);
		for (int j = 0; j < found.size(); j++)
		{
			Rect r = found[j];
			r.x *= tmpT.r;
			r.y *= tmpT.r;
			r.width *= tmpT.r;
			r.height *= tmpT.r;
			r.x += tmpT.width;
			r.y += tmpT.height;
			if ((r.width > 0) && (r.height > 0))
			{
				//rectangle(tmpMat, r.tl(), r.br(), Scalar(0, 255, 0), 1);
				rectangle(tmpMat, Point(r.x,r.y), Point(r.x+r.width,r.y+r.height), Scalar(0, 255, 0), 1);
			}
		}
		char desName[256];
		sprintf(desName, "des/%03d.png", count++);
		imwrite(desName, tmpMat);//保存文件  
		{
		imshow("test", tmpMat);
		}
		waitKey();
	}
	system("pause");
#endif
}