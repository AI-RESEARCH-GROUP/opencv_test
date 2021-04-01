#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>

#include "../include/func.h"

using namespace std;
using namespace cv;
using namespace func;

int main(int argc, char** argv)
{

    std::cout << "main.cpp" << std::endl;
    func_print();

    if (argc != 2)
    {
        cout << "Usage: opencv_test <image path>" << endl;
        return -1;
    }

    char const *imgName = "/home/zhaoqiangwei/CLionProjects/opencv_test/bin/test.jpg";
    Mat image;

    image = imread(imgName, 1);
    if (!image.data)
    {
        cout << "No image data" << endl;
        return -1;
    }
    Mat gray_img;

    cvtColor(image, gray_img, CV_BGR2GRAY);
    imwrite("/home/zhaoqiangwei/CLionProjects/opencv_test/bin/result.jpg", gray_img);

    return 0;
}