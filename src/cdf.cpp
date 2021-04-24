//
// Created by zhaoqiangwei on 2021/4/24.
//

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat src, gray, dst;

void Test(Mat &img);

int main(int argc, char *argv[]) {

    src = imread("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/image/lena.png");
    resize(src, src, Size(200, 200));
    cvtColor(src, src, COLOR_BGR2GRAY);
    Test(src);

    imwrite("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/image/lena2.png",src);
//    imshow("dst", src1);
//    waitKey(0);
}

void Test(Mat &img) {
    float gray_l[256] = {0};
    float HI[256] = {0};//图像亮度密度1/N*h(I)
    float CDF[256] = {0};//累积分布函数
    int rows = img.rows;
    int cols = img.cols;
    Mat draw_mat = Mat::zeros(600, 600, CV_8U);
    cout << rows << endl;//x

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            uchar t;
            if (img.channels() == 1) {
                t = img.at<uchar>(i, j);
                gray_l[t]++;//获得图像亮度信息
            }
        }
    }
    for (int i = 0; i < 256; i++) {

        line(draw_mat, Point(i + 150, 600 - 1), Point(i + 150, 600 - gray_l[i]), 254);
//        imshow("draw", draw_mat);
        imwrite("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/image/lena_draw.png",draw_mat);
        HI[i] = (gray_l[i] / (img.cols * img.rows * 1.0f));//获得密度概率
        cout << HI[i] << endl;
    }
    for (int i = 0; i < 255; i++) {
        if (i == 0) {
            CDF[0] = HI[0];
        } else {
            CDF[i] = (CDF[i - 1] + HI[i]);//C(I) = C(I - 1 ) +h(I)
            cout << CDF[i] << endl;
        }

    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            uchar t;
            if (img.channels() == 1) {
                t = img.at<uchar>(i, j);
                img.at<uchar>(i, j) = 255 * CDF[t];//完成图像重映射至0-255
            }
        }
        draw_mat = Mat::zeros(600, 600, CV_8U);
    }
//    imshow("yes", img);
    imwrite("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/image/lena_yes.png",img);


}