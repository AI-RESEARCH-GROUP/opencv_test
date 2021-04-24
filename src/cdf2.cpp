//
// Created by zhaoqiangwei on 2021/4/24.
//

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std ;

Mat src,gray,dst;
void global_equalization(Mat &img);
void Local_equalization(Mat &img, int size = 95, float k0 = 0.99, float k1 = 0.001, float k2 = 0.7, float E = 6);
int main(int argc, char *argv[])
{

    src =imread("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/image/stone.png");


    resize(src, src, Size(200,200));
    cvtColor(src, gray,COLOR_BGR2GRAY);
    cvtColor(src, src,COLOR_BGR2GRAY);
//    imshow("GG",src);
    Mat test = gray;
    global_equalization(src);
    Local_equalization(gray);



//    waitKey(0);
}
void global_equalization(Mat &img)
{
    float gray_l[256] = {0};
    float HI[256] = {0};//图像亮度密度1/N*h(I)
    float CDF[256] = {0};//累积分布函数
    int rows=img.rows;
    int cols=img.cols;
    Mat draw_mat = Mat::zeros(600,600,CV_8U);
    float m = 0;
    float variance = 0;
    cout << rows << endl;//x

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            uchar t;
            if(img.channels()==1)
            {
                t=img.at<uchar>(i,j);
                gray_l[t]++;//获得图像亮度信息
            }
        }
    }
    for(int i = 0; i <256; i++)
    {

        line(draw_mat,Point(i+150,600-1),Point(i+150, 600-gray_l[i]),254);
//        imshow("draw", draw_mat);
        imwrite("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/image/stone_draw.png",draw_mat);
        HI[i] = (gray_l[i]/(img.cols*img.rows*1.0f));//获得密度概率
        //cout << HI[i] << endl;
        m += HI[i]*i;
    }
    for(int i = 0; i < 256; i++)
    {
        variance += (i - m) *(i - m)*HI[i];
    }
    for(int i = 0; i < 255; i++)
    {
        if(i == 0)
        {
            CDF[0] = HI[ 0];
        }
        else
        {
            CDF[i] = (CDF[i-1] +HI[i]);//C(I) = C(I - 1 ) +h(I)
            //cout << CDF[i] << endl;
        }

    }

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            uchar t;
            if(img.channels()==1)
            {
                t=img.at<uchar>(i,j);
                img.at<uchar>(i,j) = 255*CDF[t];//完成图像重映射至0-255
            }
        } draw_mat = Mat::zeros(600,600,CV_8U);
    }
//    imshow("yes",img);
    imwrite("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/image/stone_yes.png",img);


}
void Local_equalization(Mat &img, int size , float k0 , float k1 , float k2, float E)
{
    int gray_l[256] = {0};

    float HI[256] = {0};//图像亮度密度1/N*h(I)


    float CDF[256] = {0};//累积分布函数
    int rows=img.rows;
    int cols=img.cols;
    Mat draw_mat = Mat::zeros(600,600,CV_8U);
    float global_m = 0;
    float global_variance = 0;

    int border_x = size/2;
    int border_y = size/2;
    cout << rows << endl;//x
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            uchar t;
            if(img.channels()==1)
            {
                t=img.at<uchar>(i,j);
                gray_l[t]++;//获得图像亮度信息

            }
        }
    }

    for(int i = 0; i <256; i++)
    {
        HI[i] = (gray_l[i]/(img.cols*img.rows*1.0f));//获得密度概率
        global_m += HI[i]*i;

    }

    cout << "global_m = " << global_m << endl;
    for(int i = 0; i < 256; i++)
    {
        global_variance += (i - global_m) *(i - global_m)*HI[i];
    }
    cout << "global_variance = " << global_variance << endl;
    for(int i=0; i<rows - border_x; i++)
    {
        for(int j=0; j<cols - border_y; j++)
        {
            Point center_pix = Point(j+ border_x, i + border_y);
            int local_gray_l[256] = {0};
            float local_m = 0;
            float local_variance = 0;
            float local_HI[256] = {0};
            for(int M=0; M<size; M++)
            {


                for(int N=0; N<size; N++)
                {
                    uchar t;
                    t=img.at<uchar>(i + M, j + N);
                    local_gray_l[t]++;
                }
            }

            for(int k = 0; k <256; k++)
            {
                local_HI[k] = (local_gray_l[k]/(size*size*1.0f));//获得密度概率
                local_m += local_HI[k]*k;
            }
            // cout << "local_m = " << local_m << endl;
            for(int k = 0; k < 256; k++)
            {
                local_variance += (k - local_m) *(k - local_m)*local_HI[k];
            }
//            cout << "local_variance = " << local_variance << endl;

            if( local_m < global_m*k0 && (k1 * global_variance < local_variance && local_variance < k2 * global_variance))
            {
                if(E * img.at<uchar>(center_pix) < 255)
                {
                    img.at<uchar>(center_pix) = E * img.at<uchar>(center_pix);
                    //cout << "new_pix = " << E * img.at<uchar>(center_pix) << endl;
                }
                else
                {
                    img.at<uchar>(center_pix) = 255;
                }
            }
        }
    }

//    imshow("local",img);
    imwrite("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/image/stone_local.png",img);
}