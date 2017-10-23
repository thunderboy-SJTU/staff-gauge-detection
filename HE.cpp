#include <cv.h>      //ʹ��opencv
#include <highgui.h>
#include<math.h>
#include<iostream>
#include<algorithm>
using namespace cv;
using namespace std;


Mat HE(Mat origin)   //��ֱ��ͼ��һ�����ۼӣ��ٸ���ӳ�佫����⻯
{
	double color[255];
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




int main()        //չʾͼƬ
{
	Mat srcImage = imread("F://001.png",0);
	Mat newImage = HE(srcImage);
	imwrite("F:\\0001.png", newImage);
	imshow("origin", srcImage);
	imshow("HE", newImage);
	waitKey(0);
}