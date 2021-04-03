//
// Created by zhaoqiangwei on 2021/4/3.
//

#include <opencv2/opencv.hpp>


#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int NUMBER = 100;

static Scalar randomColor( RNG& rng )
{
    int icolor = (unsigned) rng;
    return Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
}

int Drawing_Random_Lines( Mat image, char* window_name, RNG rng )
{
    int lineType = 8;
    Point pt1, pt2;
    int x_1 = 10;
    int x_2 = 100;
    int y_1 = 5;
    int y_2 = 50;
    int DELAY = 2;

    for( int i = 0; i < NUMBER; i++ )
    {
        pt1.x = rng.uniform( x_1, x_2 );
        pt1.y = rng.uniform( y_1, y_2 );
        pt2.x = rng.uniform( x_1, x_2 );
        pt2.y = rng.uniform( y_1, y_2 );

        line( image, pt1, pt2, randomColor(rng), rng.uniform(1, 10), 8 );
        imshow( window_name, image );
//        if( waitKey( DELAY ) >= 0 )
//        { return -1; }

        waitKey( DELAY );
    }
    return 0;
}

int Drawing_Random_Rectangles( Mat image, char* window_name, RNG rng ){

}

int Drawing_Random_Ellipses( Mat image, char* window_name, RNG rng ){

}

int Drawing_Random_Polylines( Mat image, char* window_name, RNG rng ){

}

int Drawing_Random_Filled_Polygons( Mat image, char* window_name, RNG rng ){

}

int Drawing_Random_Circles( Mat image, char* window_name, RNG rng ){

}

int Displaying_Random_Text( Mat image, char* window_name, RNG rng ){

}

int Displaying_Big_End( Mat image, char* window_name, RNG rng ){

}




int main(int ac, char **av) {
    RNG rng( 0xFFFFFFFF );

    int window_height = 600;
    int window_width = 600;
    char*  window_name = "test";

    /// 初始化一个0矩阵
    Mat image = Mat::zeros( window_height, window_width, CV_8UC3 );

/// 把它会知道一个窗口中
    imshow( window_name, image );

    int c = -1;
    /// 现在我们先画线
    c = Drawing_Random_Lines(image, window_name, rng);
    if( c != 0 ) return 0;

/// 继续，这次是一些矩形
    c = Drawing_Random_Rectangles(image, window_name, rng);
    if( c != 0 ) return 0;

/// 画一些弧线
    c = Drawing_Random_Ellipses( image, window_name, rng );
    if( c != 0 ) return 0;

/// 画一些折线
    c = Drawing_Random_Polylines( image, window_name, rng );
    if( c != 0 ) return 0;

/// 画被填充的多边形
    c = Drawing_Random_Filled_Polygons( image, window_name, rng );
    if( c != 0 ) return 0;

/// 画圆
    c = Drawing_Random_Circles( image, window_name, rng );
    if( c != 0 ) return 0;

/// 在随机的地方绘制文字
    c = Displaying_Random_Text( image, window_name, rng );
    if( c != 0 ) return 0;

/// Displaying the big end!
    c = Displaying_Big_End( image, window_name, rng );

}