#include "MapCylinder.h"

namespace ORB_SLAM2 {

long unsigned int MapCylinder::nNextId=0;

MapCylinder::MapCylinder(const cv::Mat &Para, Map* pMap):mnBALocalForKF(0),mnLoopClosingForKF(0),mpMap(pMap),bActive(false),bBuilt(false),mfstart(0),mfend(0),mMaxMPonAxis(-100),mMinMPonAxis(100){
    Para.copyTo(mPara);
    ToCyMatrix();
    mnId=nNextId++;
    inverseVariance=pow(39.2/mPara.at<float>(4),2);
}

MapCylinder::MapCylinder(const cv::Mat &Para):bActive(false),bBuilt(false),mfstart(0),mfend(0),mMaxMPonAxis(-100),mMinMPonAxis(100),mnBALocalForKF(0),mnLoopClosingForKF(0){
    mPara=Para;
    ToCyMatrix();
    mnId=nNextId++;
    inverseVariance=pow(39.2/mPara.at<float>(4),2);
}

MapCylinder::MapCylinder(Map* pMap):mnBALocalForKF(0),mnLoopClosingForKF(0),mpMap(pMap),bActive(false),bBuilt(false),mfstart(0),mfend(0),mMaxMPonAxis(-100),mMinMPonAxis(100){
    mPara=cv::Mat::ones(5,1,CV_32F);
    mPara.at<float>(0) = -0.0548588;
    mPara.at<float>(1) = 0.0358857;
    mPara.at<float>(2) = 1.94072;
    mPara.at<float>(3) = 0.174347;
    mPara.at<float>(4) = 0.447632;
    mnId=nNextId++;
}

void MapCylinder::SetBadFlag(){
    for(list<MapPoint*>::iterator lit=mlpCyMapPoints.begin(), lend=mlpCyMapPoints.end(); lit!=lend; lit++){
        (*lit)->mpCylinder=static_cast<MapCylinder*>(NULL);
    }
    for(list<KeyFrame*>::iterator lit=mlpCylindricalKF.begin() , lend=mlpCylindricalKF.end(); lit!=lend; lit++){
        (*lit)->AddMapCylinder(NULL);
    }
    mpMap->EraseMapCylinder(this);
}

void MapCylinder::AddCyMapPoint(MapPoint *point){
    point->SetCylinder(this);
    cv::Mat pw = point->GetWorldPos();
    cv::Mat pcy = mRcyw * pw + mPcyw;
    if(mMaxMPonAxis < pcy.at<float>(2)){
        mMaxMPonAxis = pcy.at<float>(2);
    }
    if(mMinMPonAxis > pcy.at<float>(2)){
        mMinMPonAxis = pcy.at<float>(2);
    }
    // 有些点可能会被删除
    mlMPsOnAxis.push_back(pcy.at<float>(2));
    mlpCyMapPoints.push_back(point);
}
void MapCylinder::ToCyMatrix(){
    // cv::Mat rotationVector = mPara(cv::Rect(0, 0, 3, 1)).clone();
    // 将子矩阵转换为旋转矩阵
    // cv::Mat rotationMat;
    Eigen::Matrix<double, 5, 1> param = Converter::toParam5d(mPara);
    Eigen::Matrix3d Rcyw = Eigen::AngleAxisd(param.block<3,1>(0,0).norm(), param.block<3,1>(0,0).normalized()).toRotationMatrix();
    Eigen::Matrix3d Rwcy = Rcyw.transpose();
    Eigen::Vector3d Pcyw = Eigen::Vector3d(param(3), 0, 0);
    Eigen::Vector3d Pwcy = -Rwcy * Pcyw;
    mTcyw = Converter::toCvSE3(Rcyw, Pcyw);
    mRcyw = Converter::toCvMat(Rcyw);
    mPcyw = Converter::toCvMat(Pcyw);
    mTwcy = Converter::toCvSE3(Rwcy, Pwcy);
    mRwcy = Converter::toCvMat(Rwcy);
    mPwcy = Converter::toCvMat(Pwcy);
}

void MapCylinder::ToCyPara(){
    Eigen::Matrix<double, 4, 4> Twcy = Converter::toMatrix4d(mTwcy);
    Eigen::Matrix3d Rwcy = Twcy.block<3,3>(0, 0);
    Eigen::Matrix3d Rcyw = Rcyw.transpose();
    Eigen::Vector3d Pwcy = Twcy.block<3,1>(0, 3);
    Eigen::Vector3d Pcyw = -Rwcy * Pcyw;
    mTcyw = Converter::toCvSE3(Rcyw, Pcyw);
    mRcyw = Converter::toCvMat(Rcyw);
    mPcyw = Converter::toCvMat(Pcyw);
    mTwcy = Converter::toCvSE3(Rwcy, Pwcy);
    mRwcy = Converter::toCvMat(Rwcy);
    mPwcy = Converter::toCvMat(Pwcy);

    Eigen::AngleAxisd rotationAngleAxis(Rwcy);
    Eigen::Matrix<double, 5, 1> param = Eigen::Matrix<double, 5, 1>::Zero();
    param.block<3, 1>(0, 0) = rotationAngleAxis.axis() * rotationAngleAxis.angle();
    param(3) = Twcy(3, 1);
    param(4) = mPara.at<float>(4);
    mPara = Converter::toCvMat(param);
}


void MapCylinder::CalculateLength(){
    const int division = 50;
    std::vector<int> Hist(division, 0);
    const int thr = 100;
    const double factor = (mMaxMPonAxis - mMinMPonAxis) / division;
    // std::cout << "factor = " << factor << std::endl;
    for(std::list<double>::iterator sit = mlMPsOnAxis.begin(), send = mlMPsOnAxis.end(); sit != send; sit++){
        float it = *sit;
        int index = static_cast<int>((it - mMinMPonAxis) / factor);
        if(index == division) index--;//处理遍历到mMaxMPonAxis的情况
        Hist[index]++;
    }
    for(int i = 0; i < division; i++){
        if(Hist[i] > thr){
            mfstart = mMinMPonAxis + i * factor;
            break;
        }
    }
    for(int i = division - 1; i >= 0; i--){
        if(Hist[i] > thr){
            mfend = mMinMPonAxis + i * factor;
            break;
        }
    }
}

// LCTODO 判断mappoint是否在圆柱面上
bool MapCylinder::OnCylinder(MapPoint* pMP){
    // 圆柱投影生成的点
    if(pMP->mpCylinder == this){
        return true;
    }
    // 点面距离判断
    cv::Mat Pw = pMP->GetWorldPos();
    cv::Mat Pcy = mRcyw * Pw + mPcyw;
    float distance = abs(GetRadius() - sqrt(Pcy.at<float>(0) * Pcy.at<float>(0) + Pcy.at<float>(1) * Pcy.at<float>(1)));
    if(distance < 0.04 * GetRadius()){
        return true;
    }

    return false;
}
}

    