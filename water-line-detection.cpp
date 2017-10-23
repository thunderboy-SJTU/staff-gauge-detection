#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h> 
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>

using namespace cv;
using namespace std;


void OnDrawDotline(CvPoint s, CvPoint d, IplImage *img)
{
	CvPoint pa, pb;

	double k = (s.y - d.y) / (s.x - d.x + 0.000001);

	double h = img->height;
	double w =img->width;

	pa.x = w;
	pa.y = s.y + k*(w - s.x);
	pb.y = d.y - k*d.x;
	pb.x = 0;

	cvLine(img, pb, pa, CV_RGB(255, 0, 0), 3, CV_AA, 0);  

}

int main()
{
	const char* filename = "F:/7.jpg";
	IplImage* src = cvLoadImage(filename, 0);
	IplImage* dst;
	IplImage* color_dst;
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* lines = 0;
	int i;
	if (!src)
	{
		return -1;
	}
	dst = cvCreateImage(cvGetSize(src), 8, 1);
	color_dst = cvCreateImage(cvGetSize(src), 8, 3);
	cvCanny(src, dst, 50, 200, 3);
	cvCvtColor(dst, color_dst, CV_GRAY2BGR);
	int maxlength = 0;
	CvPoint* maxline = NULL;
	lines = cvHoughLines2(dst, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI / 180, 100, 100, 15);
	for (i = 0; i < lines->total; i++)
	{
		CvPoint* line = (CvPoint*)cvGetSeqElem(lines, i);
		double k = (line[1].y -line[0].y) / (line[1].x - line[0].x + 0.000001);
		if (k >= 0.3|| k <= -0.3)
			continue;
		if (sqrt((line[0].x - line[1].x)*(line[0].x - line[1].x) + (line[0].y - line[1].y)*(line[0].y - line[1].y)) >= maxlength) {
			maxlength = sqrt((line[0].x - line[1].x)*(line[0].x - line[1].x) + (line[0].y - line[1].y)*(line[0].y - line[1].y));
			maxline = line;
		}
	}
	OnDrawDotline(maxline[0], maxline[1], color_dst);
	cvNamedWindow("Source", 1);
	cvShowImage("Source", src);
	cvNamedWindow("Hough", 1);
	cvShowImage("Hough", color_dst);
	cvWaitKey(0);

	return 0;
}