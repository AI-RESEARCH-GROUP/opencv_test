//
// Created by zhaoqiangwei on 2021/4/4.
//

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

int main(int ac, char **argv) {
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }
//输入图像
    Mat image = imread("E:\\paper_and_code\\20180906\\SURF_test_1008\\1.jpg");

    Mat outimage;//输出图像
    vector<KeyPoint> keyPoints;//关键点
    Mat dstImage;

    Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(minHessian);//设置SURF特征检测器
    detector->detectAndCompute(image, Mat(), keyPoints, dstImage);//检测图像中的SURF特征点


//// detecting keypoints
//    SurfFeatureDetector detector(400);
//    vector<KeyPoint> keypoints1, keypoints2;
//    detector.detect(img1, keypoints1);
//    detector.detect(img2, keypoints2);
//
//// computing descriptors
//    SurfDescriptorExtractor extractor;
//    Mat descriptors1, descriptors2;
//    extractor.compute(img1, keypoints1, descriptors1);
//    extractor.compute(img2, keypoints2, descriptors2);
//
//// matching descriptors
//    BruteForceMatcher<L2<float> > matcher;
//    vector<DMatch> matches;
//    matcher.match(descriptors1, descriptors2, matches);

// drawing the results
    namedWindow("matches", 1);
    Mat img_matches;
//    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("matches", img_matches);
    waitKey(0);
}