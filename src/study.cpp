//
// Created by zhaoqiangwei on 2021/4/1.
//


#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>


#include<iostream>

using namespace std;
using namespace cv;

Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));

    int channels = I.channels();

    int nRows = I.rows * channels;
    int nCols = I.cols;

    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }

    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];
        }
    }
    return I;
}


int func6(){
    /// 读入用户提供的图像
    std::string image_path = "";
    std::cout<<"* Enter image-path: ";
    std::cin>> image_path;
    Mat image = imread( image_path);
    Mat new_image = Mat::zeros( image.size(), image.type() );

    /// 初始化
    double alpha = 0.5; double beta; double input;
    cout << " Basic Linear Transforms " << endl;
    cout << "-------------------------" << endl;
    cout << "* Enter the alpha value [1.0-3.0]: ";
    cin >> alpha;
    cout << "* Enter the beta value [0-100]: ";
    cin >> beta;

    /// 执行运算 new_image(i,j) = alpha*image(i,j) + beta
    for( int y = 0; y < image.rows; y++ )
    {
        for( int x = 0; x < image.cols; x++ )
        {
            for( int c = 0; c < 3; c++ )
            {
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
            }
        }
    }

    /// 创建窗口
    namedWindow("Original Image", 1);
    namedWindow("New Image", 1);

    /// 显示图像
    imshow("Original Image", image);
    imshow("New Image", new_image);

    /// 等待用户按键
    waitKey();
    return 0;
}

int func5(){
    double alpha = 0.5; double beta; double input;

    Mat src1, src2, dst;

    /// Ask the user enter alpha
    std::cout<<" Simple Linear Blender "<<std::endl;
    std::cout<<"-----------------------"<<std::endl;
    std::cout<<"* Enter alpha [0-1]: ";
    std::cin>>input;

    /// We use the alpha provided by the user iff it is between 0 and 1
    if( alpha >= 0 && alpha <= 1 )
    { alpha = input; }

    /// Read image ( same size, same type )
    src1 = imread("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/img1.jpg");
    src2 = imread("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/img2.jpg");

    if( !src1.data ) { printf("Error loading src1 \n"); return -1; }
    if( !src2.data ) { printf("Error loading src2 \n"); return -1; }

    /// Create Windows
    namedWindow("Linear Blend", 1);

    beta = ( 1.0 - alpha );
    addWeighted( src1, alpha, src2, beta, 0.0, dst);

    imshow( "Linear Blend", dst );

    waitKey(0);
    return 0;
}

int func4(){
    Mat src1 = imread("../../images/LinuxLogo.jpg");

    int divideWith; // convert our input string to number - C++ style
    stringstream s;
    s << "10";
    s >> divideWith;
    if (!s)
    {
        cout << "Invalid number entered for dividing. " << endl;
        return -1;
    }

    uchar table[256];
    for (int i = 0; i < 256; ++i)
        table[i] = divideWith* (i/divideWith);

    cout << "Times passed in seconds: " << table << endl;

    double t = (double)getTickCount();
// 做点什么 ...
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Times passed in seconds: " << t << endl;

}

int func0() {

    Mat M2(2,2, CV_8UC3, Scalar(0,0,255));
    cout << "M = " << endl << " " << M2 << endl << endl;

    int sz[3] = {2,2,2};
    Mat L(3,sz, CV_8UC(1), Scalar::all(0));

    M2.create(4,4, CV_8UC(2));
    cout << "M = "<< endl << " "  << M2 << endl << endl;

    Mat C2 = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cout << "C = " << endl << " " << C2 << endl << endl;

    Mat RowClone = C2.row(1).clone();
    cout << "RowClone = " << endl << " " << RowClone << endl << endl;

    Mat R = Mat(3, 2, CV_8UC3);
    randu(R, Scalar::all(0), Scalar::all(255));

    cout << "R (default) = " << endl <<        R           << endl << endl;
    cout << "R (python)  = " << endl << format(R, Formatter::FMT_PYTHON) << endl << endl;
    cout << "R (csv)     = " << endl << format(R, Formatter::FMT_CSV ) << endl << endl;

    Point2f P(5, 1);
    cout << "Point (2D) = " << P << endl << endl;
    Point3f P3f(2, 6, 7);
    cout << "Point (3D) = " << P3f << endl << endl;

    vector<float> v;
    v.push_back( (float)CV_PI);
    v.push_back(2);
    v.push_back(3.01f);
    cout << "Vector of floats via Mat = " << Mat(v) << endl << endl;


    Mat M(2,2, CV_8UC3, Scalar(0,0,255));
    cout << "M = " << endl << " " << M << endl << endl;

    cv::Mat a(100, 100, CV_32F);
    cv::randu(a, cv::Scalar::all(1), cv::Scalar::all(std::rand()));
    cv::log(a, a);
    a /= std::log(2.);
    std::cout << a << std::endl;

    // create a big 8Mb matrix
    cv::Mat A(1000, 1000, CV_64F);
// create another header for the same matrix;
// this is an instant operation, regardless of the matrix size.
    cv::Mat B = A;
// create another header for the 3-rd row of A; no data is copied either
    cv::Mat C = B.row(3);
// now create a separate copy of the matrix
    cv::Mat D = B.clone();
// copy the 5-th row of B to C, that is, copy the 5-th row of A
// to the 3-rd row of A.
    B.row(5).copyTo(C);
// now let A and D share the data; after that the modified version
// of A is still referenced by B and C.
    A = D;
// now make B an empty matrix (which references no memory buffers),
// but the modified version of A will still be referenced by C,
// despite that C is just a single row of the original A
    B.release();
// finally, make a full copy of C. As a result, the big modified
// matrix will be deallocated, since it is not referenced by anyone
    C = C.clone();
}

int func1() {
    VideoCapture cap(0);
    if(!cap.isOpened()) return -1;
    Mat frame, edges;
    namedWindow("edges",1);
    for(;;)
    {
        cap >> frame;
        cvtColor(frame, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        Canny(edges, edges, 0, 30, 3);
        imshow("edges", edges);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}

int func2(){
//    if ( argc != 2 )
//    {
//        printf("usage: DisplayImage.out <Image_Path>\n");
//        return -1;
//    }
    Mat image;
    char const *imgName = "/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/test.jpg";
    image = imread( imgName, 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);
    return 0;
}
//
//Mat Flannfeaturecompare(Mat srcImage1, Mat srcImage2)
//{
//    int hessPara = 400;
//    Ptr<Feature2D> sift = SIFT::create();
//    Ptr<SURF> detector = cv::  SURF::create(hessPara);
//    vector<KeyPoint> keypoints1, keypoints2;
//    Mat descriptors1, descriptors2;
//    detector->detectAndCompute(srcImage1, Mat(), keypoints1, descriptors1);
//    detector->detectAndCompute(srcImage2, Mat(), keypoints2, descriptors2);
//    FlannBasedMatcher matcher;
//    //FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
//    vector<DMatch> matches;
//    //BFMatcher matcher;
//    matcher.match(descriptors1, descriptors2, matches);
//    double max_dist = 0;
//    double min_dist = 1000;
//    //距离判断--最优匹配点
//    for (int i = 0; i < descriptors1.rows; i++)
//    {
//        double dist = matches[i].distance;
//        if (dist < min_dist)
//            min_dist = dist;
//        if (dist > max_dist)
//            max_dist = dist;
//    }
//    cout << "max_dist=" << max_dist << endl << "min_dist=" << min_dist << endl;
//    //最佳匹配点
//    vector<DMatch> matchVec;
//    for (int i = 0; i < descriptors1.rows; i++)
//    {
//        if (matches[i].distance < 5 * min_dist)
//        {
//            matchVec.push_back(matches[i]);     //push_back 将满足条件的值赋值给matchVec数组
//        }
//    }
//    Mat matchMat, matchMat2;
//    drawMatches(srcImage1, keypoints1, srcImage2, keypoints2, matchVec, matchMat, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//    //imshow("matchMat", matchMat);
//    //特征点一致性检测
//    vector<Point2f> obj, scene;
//    for (int i = 0; i < matchVec.size(); i++)
//    {
//        obj.push_back(keypoints1[matchVec[i].queryIdx].pt);   //为前图进行特征点索引  pt代表point(pt.x,pt.y);
//        scene.push_back(keypoints2[matchVec[i].trainIdx].pt);  //为模板进行特征点索引
//    }
//    Mat H = findHomography(obj, scene, RANSAC);  //随机点匹配
//    vector<Point2f> objCorner(4), sceneCors(4);
//    objCorner[0] = Point(0, 0);
//    objCorner[1] = Point(srcImage1.cols, 0);
//    objCorner[2] = Point(srcImage1.cols, srcImage1.rows);
//    objCorner[3] = Point(0, srcImage1.rows);
//    perspectiveTransform(objCorner, sceneCors, H);  //映射矩阵
//    Point2f offset((float)srcImage1.cols, 0);   //偏移量的增加
//    //line(matchMat, sceneCors[0] + offset, sceneCors[1] + offset, Scalar(0, 255, 0), 2);
//    //line(matchMat, sceneCors[1] + offset, sceneCors[2] + offset, Scalar(0, 255, 0), 2);
//    //line(matchMat, sceneCors[2] + offset, sceneCors[3] + offset, Scalar(0, 255, 0), 2);
//    //line(matchMat, sceneCors[3] + offset, sceneCors[0] + offset, Scalar(0, 255, 0), 2);
//    float min_x = 1000, min_y = 1000;
//    float max_x = 0, max_y = 0;
//    for (int i = 0; i < 4; i++)
//    {
//        if (sceneCors[i].x < min_x)
//            min_x = sceneCors[i].x;
//        if (sceneCors[i].y < min_y)
//            min_y = sceneCors[i].y;
//        for (int j = i; j < 3; j++)
//        {
//            float max_dis_x = abs(sceneCors[i].x - sceneCors[j + 1].x);
//            if (max_dis_x > max_x)
//                max_x = max_dis_x;
//            float max_dis_y = abs(sceneCors[i].y - sceneCors[j + 1].y);
//            if (max_dis_y > max_y)
//                max_y = max_dis_y;
//        }
//
//    }
//    //通过两张图片进行对比特征点匹配
//    //添加偏移量区域图的横向坐标偏移
//    rectangle(matchMat, Rect(min_x + srcImage1.cols, min_y, max_x, max_y), Scalar(0, 0, 255), 2, 8, 0);
//    Mat dst = srcImage2.clone();
//    rectangle(dst,
//              Rect(sceneCors[0].x, sceneCors[0].y, sceneCors[1].x - sceneCors[0].x, sceneCors[3].y - sceneCors[0].y),
//              Scalar(0, 0, 255), 2, 8, 0);
//    imshow("ObjectMat", matchMat);
//    imshow("dst", dst);
//    return matchMat;
//}
//
//int func3(){
//    //载入源图像，显示并转换灰度图
//    Mat srcImage=imread("/Users/new/Desktop/5.jpg"),grayImage;
//    namedWindow("image[origin]",1);
//    imshow("image[origin]",srcImage);
//    cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
//
//    //检测surf关键点、提取训练图像描述符
//    vector<KeyPoint> keyPoint;
//    Mat descriptor;
//    SurfFeatureDetector featureDetector(80);
//    featureDetector.detect(grayImage,keyPoint);
//    SurfDescriptorExtractor featureExtractor;
//    featureExtractor.compute(grayImage, keyPoint, descriptor);
//
//    //创建基于FLANN的描述符匹配对象
//    FlannBasedMatcher matcher;
//    vector<Mat> desc_collection(1,descriptor);
//    matcher.add(desc_collection);
//    matcher.train();
//
//    //创建视频对象、定义帧率
//    VideoCapture cap(0);
//    unsigned int frameCount=0;//帧数
//
//    //不断循环，直到q键被按下
//    while(char(waitKey(1))!=27)
//    {
//        //参数设置
//        int64 time0=getTickCount();
//        Mat testImage,grayImage_test;
//        cap>>testImage;//采集视频到testImage中
//        if(testImage.empty())
//            continue;
//        //转换图像到灰度
//        cvtColor(testImage, grayImage_test, COLOR_BGR2GRAY);
//        //检测S关键点、提取测试图像描述符
//        vector<KeyPoint> keyPoint_test;
//        Mat descriptor_test;
//        featureDetector.detect(grayImage_test, keyPoint_test);
//        featureExtractor.compute(grayImage_test, keyPoint_test, descriptor_test);
//
//        //匹配训练和测试描述符
//        vector<vector<DMatch>>matches;
//        matcher.knnMatch(descriptor_test, matches, 2);
//        //根据劳氏算法，得到优秀的匹配点
//        vector<DMatch> goodMatches;
//        for(unsigned int i=0;i<matches.size();++i)
//        {
//            if(matches[i][0].distance<0.6*matches[i][1].distance)
//                goodMatches.push_back(matches[i][0]);
//        }
//        //绘制匹配点并显示窗口
//        Mat dstImage;
//        drawMatches(testImage, keyPoint_test, srcImage, keyPoint, goodMatches, dstImage);
//        namedWindow("image[match]",1);
//        imshow("image[match]",dstImage);
//
//        //输出帧率信息
//        cout<<"当前帧率为："<<getTickFrequency()/(getTickCount()-time0)<<endl;
//    }
//
//}

int main() {
//    func0();
//    func1();
//    func2();
//    func4();
//    func5();
    func6();
    return 0;
}