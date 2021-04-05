//
// Created by zhaoqiangwei on 2021/4/4.
//

//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <iostream>
//#include <opencv2/imgproc/types_c.h>
//#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
//
//using namespace cv;
//using namespace std;

#include <iostream>
//#include <opencv2/opencv.hpp>

#include "/home/zhaoqiangwei/soft/opencv/installed/opencv4/include/opencv4/opencv2/opencv.hpp"
#include "/home/zhaoqiangwei/soft/opencv/installed/opencv4/include/opencv4/opencv2/xfeatures2d/nonfree.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;


int surf_1img( int argc, char* argv[] ){
    CommandLineParser parser( argc, argv, "{@input | box.jpg | input image}" );
    Mat src = imread( parser.get<String>( "@input" ), IMREAD_GRAYSCALE );
    if ( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints;
    detector->detect( src, keypoints );
    //-- Draw keypoints
    Mat img_keypoints;
    drawKeypoints( src, keypoints, img_keypoints );
    //-- Show detected (drawn) keypoints
    imshow("SURF Keypoints", img_keypoints );
    waitKey();
    return 0;
}


int match_2img(int ac, char **argv) {
    cv::Mat imageL = cv::imread("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/img2.jpg");
    cv::Mat imageR = cv::imread("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/img3.jpg");

    //提取特征点方法
    //SIFT
//    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
//    cv::Ptr<cv::SIFT> sift = cv::SIFT::Creat(); //OpenCV 4.4.0 及之后版本
    //ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    //SURF
//    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();

    //特征点
    std::vector<cv::KeyPoint> keyPointL, keyPointR;
    //单独提取特征点
    orb->detect(imageL, keyPointL);
    orb->detect(imageR, keyPointR);

    //画特征点
    cv::Mat keyPointImageL;
    cv::Mat keyPointImageR;
    drawKeypoints(imageL, keyPointL, keyPointImageL, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(imageR, keyPointR, keyPointImageR, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //显示窗口
    cv::namedWindow("KeyPoints of imageL");
    cv::namedWindow("KeyPoints of imageR");

    //显示特征点
    cv::imshow("KeyPoints of imageL", keyPointImageL);
    cv::imshow("KeyPoints of imageR", keyPointImageR);

    //特征点匹配
    cv::Mat despL, despR;
    //提取特征点并计算特征描述子
    orb->detectAndCompute(imageL, cv::Mat(), keyPointL, despL);
    orb->detectAndCompute(imageR, cv::Mat(), keyPointR, despR);

    //Struct for DMatch: query descriptor index, train descriptor index, train image index and distance between descriptors.
    //int queryIdx –>是测试图像的特征点描述符（descriptor）的下标，同时也是描述符对应特征点（keypoint)的下标。
    //int trainIdx –> 是样本图像的特征点描述符的下标，同样也是相应的特征点的下标。
    //int imgIdx –>当样本是多张图像的话有用。
    //float distance –>代表这一对匹配的特征点描述符（本质是向量）的欧氏距离，数值越小也就说明两个特征点越相像。
    std::vector<cv::DMatch> matches;

    //如果采用flannBased方法 那么 desp通过orb的到的类型不同需要先转换类型
    if (despL.type() != CV_32F || despR.type() != CV_32F)
    {
        despL.convertTo(despL, CV_32F);
        despR.convertTo(despR, CV_32F);
    }

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    matcher->match(despL, despR, matches);

    //计算特征点距离的最大值
    double maxDist = 0;
    for (int i = 0; i < despL.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist > maxDist)
            maxDist = dist;
    }

    //挑选好的匹配点
    std::vector< cv::DMatch > good_matches;
    for (int i = 0; i < despL.rows; i++)
    {
        if (matches[i].distance < 0.5*maxDist)
        {
            good_matches.push_back(matches[i]);
        }
    }



    cv::Mat imageOutput;
    cv::drawMatches(imageL, keyPointL, imageR, keyPointR, good_matches, imageOutput);

    cv::namedWindow("picture of matching");
    cv::imshow("picture of matching", imageOutput);
    imwrite("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/bin/result.jpg", imageOutput);

    cv::waitKey();
    return 0;

}

int main( int argc, char* argv[] ){
    int result = 0;
//    result = surf_1img(argc,argv);
    result = match_2img(argc,argv);

    return result;
}