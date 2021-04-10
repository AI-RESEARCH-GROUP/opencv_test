#include <opencv2/opencv.hpp>

#include<sstream>
#include<string>

using namespace cv;
using namespace std;

string xml_dir = "/usr/share/opencv/haarcascades/";

vector<Rect> detectFaces(Mat img_gray){
    CascadeClassifier faces_cascade;
    faces_cascade.load(xml_dir + "haarcascade_frontalface_alt.xml");
    vector<Rect> faces;
    faces_cascade.detectMultiScale(img_gray,faces,1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    return faces;
}

void drawFaces(Mat img,vector<Rect> faces){
    namedWindow("draw faces");
    for(size_t i=0;i<faces.size();i++){
        //?????????????????????,?????????????
        Point center(faces[i].x + faces[i].width/2,faces[i].y + faces[i].height/2);
        ellipse(img,center,Size(faces[i].width/2,faces[i].height/1.5),0,0,360,Scalar(0,255,0),2,8,0);
    }
    imshow("draw faces",img);
}

void saveFaces(Mat img,Mat img_gray){
    vector<Rect> faces = detectFaces(img_gray);
    for(size_t i=0; i<faces.size();i++){
        stringstream buffer;
        buffer<<i;
        string saveName = "faces/"+ buffer.str() + ".jpg";
        Rect roi = faces[i];
        imwrite(saveName,img(roi));
    }
}

void detectDrawEyes(Mat img,Mat img_gray){
    vector<Rect> faces = detectFaces(img_gray);
    for(size_t i=0; i<faces.size();i++){
        Mat faceROI = img_gray(faces[i]);
        CascadeClassifier eyes_cascade;
        eyes_cascade.load(xml_dir + "haarcascade_eye.xml");
        vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        for(size_t j=0;j<eyes.size();j++){
            Point eyes_center(faces[i].x+eyes[j].x+eyes[j].width/2,faces[i].y+eyes[j].y+eyes[j].height/2);
            int r = cvRound((eyes[j].width + eyes[j].height)*0.25);
            circle(img,eyes_center,r,Scalar(255,0,0),1,8,0);
        }
    }
    namedWindow("detect and draw eyes");
    imshow("detect and draw eyes",img);
}

int main(){
    Mat img = imread("/home/zhaoqiangwei/mygit/com.cplusplus/opencv_test/obama.jpg");
    Mat img_gray;
    cvtColor(img,img_gray,COLOR_BGR2GRAY );
    equalizeHist(img_gray,img_gray);
    vector<Rect> faces = detectFaces(img_gray);
    saveFaces(img,img_gray);
    drawFaces(img,faces);
    detectDrawEyes(img,img_gray);
    waitKey(0);
    return 0;
}
