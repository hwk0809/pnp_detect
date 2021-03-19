#include <ros/ros.h>
#include <iostream>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "sensor_msgs/Image.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pnp_detect/points.h>
#include <geometry_msgs/Vector3Stamped.h>


using namespace cv;
using namespace std;



cv_bridge::CvImage cv_bridge_current_img{};
Mat cur_img;
float Bottom_Servo;
float Top_Servo;
int Servo_time;

class Board{
    //左上角开始 顺时针 前x 左y 上z


    std::vector<cv::Point3f> initBoardPoints(){
        objectPts.emplace_back(10, 55, -150);
        objectPts.emplace_back(50, 55, -150);
        objectPts.emplace_back(50, -55, -150);
        objectPts.emplace_back(10, -55, -150);
        return objectPts;
    }



public:
    //板子上的四个点
    std::vector<cv::Point3f> objectPts;
    Board(){
        initBoardPoints();
    }
};

//回调函数，cvbridge
void imageCB(const sensor_msgs::Image msg){
    static int count = 0;
    count++;
//    ROS_INFO_STREAM_THROTTLE(0.2,"new msg in !! nums :  " << count);
    //    cv_bridge::CvImagePtr cv;
    auto cv = cv_bridge::toCvCopy(msg,"mono8");
    cv_bridge_current_img.image = cv->image;
    cv_bridge_current_img.encoding = cv->encoding;
    cv_bridge_current_img.header = cv->header;
    cur_img = cv->image;
}

//回调函数，
void imageCB2(const geometry_msgs::Vector3Stamped msg2){
    Servo_time=msg2.header.stamp.sec;
    Bottom_Servo=msg2.vector.x;
    Top_Servo=msg2.vector.y;
}

//调试，画出树状结构
void debug_print_hierarchy( const vector<Vec4i> & hierarchy,const vector<vector<Point>>  & contours){

    for(int i = 0;i<hierarchy.size();i++){
        cout << "ID: "<< i<< " hierarchy： ";
        for (int j = 0; j < 4; ++j) {
            cout << hierarchy[i][j]<<" ";
        }
        cout << " point : ";
        for (int j = 0; j < contours[i].size(); ++j) {
            cout << contours[i][j]<<" ";
        }
        cout << endl;
    }
}

//找中间层节点
vector<int> find_mid_id(const vector<Vec4i> & hierarchy){
    vector<int> res;
    for (int i = 0; i < hierarchy.size(); ++i) {
        if(hierarchy[i][2]!=-1&&hierarchy[i][3]!=-1){
            res.push_back(i);
        }
    }
    return res;
}

//找内层节点
vector<int> find_inner_id(const vector<Vec4i> & hierarchy,const vector<int> & mid_id){
    vector<int> res;
    vector<int> res_one;
    vector<int> res_two;
    if(mid_id.size()!=2)return res;
    for (int i = 0; i < hierarchy.size(); ++i) {
        if(hierarchy[i][3]==mid_id[0]){
            res_one.push_back(i);
        }
        if(hierarchy[i][3]==mid_id[1]){
            res_two.push_back(i);
        }
        if(res_one.size()==3&&res_two.size()==1){
            res_two.insert(res_two.end(),res_one.begin(),res_one.end());
            return res_two;
        }else if(res_one.size()==1&&res_two.size()==3){
            res_one.insert(res_one.end(),res_two.begin(),res_two.end());
            return res_one;
        }
    }
    return res;
}
bool myCom(const Point2i& a,const Point2i& b){
    return a.x<b.x;
}

vector<Point2f> find_point_center(const vector<vector<Point>> & contours,const vector<int> & inner_id){
    vector<Point2f> res;
    vector<Point2f> new_res;
    for (int i = 0; i < inner_id.size(); ++i) {
        if(contours.size()<=inner_id[i])return vector<Point2f>();
        auto M = moments(contours[inner_id[i]]);
        Point2f center;
        center.x = M.m10/M.m00;
        center.y = M.m01/M.m00;
        res.push_back(center);
    }
    new_res.push_back(res[0]);
    Point2f vec10(res[1]-res[0]);
    Point2f vec20(res[2]-res[0]);
    Point2f vec30(res[3]-res[0]);
    Point2f vec_base(0,1);
    auto angle10 = 1000*acos(vec10.dot(vec_base)/sqrt(vec10.x*vec10.x+vec10.y*vec10.y));
    auto angle20 = 1000*acos(vec20.dot(vec_base)/sqrt(vec20.x*vec20.x+vec20.y*vec20.y));
    auto angle30 = 1000*acos(vec30.dot(vec_base)/sqrt(vec30.x*vec30.x+vec30.y*vec30.y));
    vector<Point2i> angles {Point2i(angle10,1),Point2i (angle20,2),Point2i (angle30,3)};
    sort(angles.begin(),angles.end(),myCom);
    for (int i = 0; i < angles.size(); ++i) {
        new_res.push_back(res[angles[i].y]);
    }
    return new_res;
}

int main(int argc,char* *argvs){
    ros::init(argc,argvs,"detetc");
    ros::NodeHandle nh;
    pnp_detect::points pnp_points_msg;
    ros::Subscriber image_sub = nh.subscribe("/MVcam/image",1,imageCB);

    ros::Subscriber servo_sub = nh.subscribe("/servo/position",1,imageCB2);//only for rotate

    ros::Publisher pnp_publish = nh.advertise<pnp_detect::points>("/image_pnp_points",1);
    ros::Rate loop_rate(100);
    Board uav_board;
    static cv::Mat cameraMatrix;
    static cv::Mat rvec, tvec;
    cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
    static const std::vector<double> distcoeffs{-0.0622484869016584, 0.116141620134878, 0.0, 0.0, 0.0};
    cameraMatrix.at<double>(0, 0) = 1684.659238683180;
    cameraMatrix.at<double>(0, 2) = 656.480920681900;
    cameraMatrix.at<double>(1, 1) = 1684.163794939990;
    cameraMatrix.at<double>(1, 2) = 535.952107137916;
    cameraMatrix.at<double>(2, 2) = 1;
    while(ros::ok()){
        ros::spinOnce();
        if(cur_img.empty())continue;

        cv::threshold(cur_img, cur_img, 100, 255, cv::THRESH_TOZERO);
        cv::threshold(cur_img, cur_img, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        imshow("lijiayi_bag", cur_img);
        waitKey(2);
        //
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(cur_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        Mat resultImage = Mat ::zeros(cur_img.size(),CV_8U);
        drawContours(resultImage, contours, -1, Scalar(255, 0, 255));
//        imshow("result", resultImage);
        auto mid_id = find_mid_id(hierarchy);
        if(mid_id.size()!=2) continue;
        auto inner_id = find_inner_id(hierarchy,mid_id);
        if(inner_id.size()!=4) continue;
        auto point_center = find_point_center(contours,inner_id);
        if(point_center.size()!=4) continue;
        for (int i = 0; i < point_center.size(); ++i) {
            cv::putText(resultImage, std::to_string(i) , point_center[i], cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255,120,0), 3);
        }
        pnp_points_msg.header.stamp = ros::Time::now();
        pnp_points_msg.header.stamp.sec = Servo_time;
        for (int i = 0; i < point_center.size(); ++i) {
            pnp_points_msg.PNPPoints[i].x = point_center[i].x;
            pnp_points_msg.PNPPoints[i].y = point_center[i].y;
        }

        pnp_points_msg.PNPPoints[4].x=Bottom_Servo;
        pnp_points_msg.PNPPoints[4].y=Top_Servo;

        pnp_publish.publish(pnp_points_msg);
        cv::solvePnP(uav_board.objectPts,point_center,cameraMatrix,distcoeffs,rvec,tvec,false, SOLVEPNP_ITERATIVE);
        cout <<"T = " << tvec << "\n";
        imshow("resultImage", resultImage);
        waitKey(2);
    }
}

