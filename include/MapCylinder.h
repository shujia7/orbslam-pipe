#ifndef ORB_SLAM2_MAPCYLINDER_H
#define ORB_SLAM2_MAPCYLINDER_H

#include "Map.h"
#include "MapPoint.h"
#include "Converter.h"
#include <unordered_set>
#include<opencv2/core/core.hpp>
#include <list>
#include<opencv2/calib3d/calib3d.hpp>
namespace ORB_SLAM2
{
class Map;
class KeyFrame;
// class MapPoint;
class MapPoint;
class MapCylinder{
public:
    MapCylinder(const cv::Mat &Para, Map* pMap);
    MapCylinder(const cv::Mat &Para);
    MapCylinder(Map *pMap);

    cv::Mat GetWPara(){
        return mPara.clone();
    }
    // 准备圆柱初始值
    void cyPreparation(const cv::Mat &Para){
        mPara = Para.clone();
    }
    // 更改圆柱参数
    void SetWorldPara(const cv::Mat &Para){
        mPara = Para.clone();
        ToCyMatrix();
    }

    void SetWorldMat(const cv::Mat &Mat){
        mTwcy = Mat.clone();
        ToCyPara();
    }
    // 储存圆柱帧
    void AddCylindricalKF(KeyFrame * kF){
        msCylindricalKF.insert(kF);
    }

    void AddCyMapPoint(MapPoint *point);

    void AddPastMapPoint(MapPoint *point){
        mlpPastMapPoints.push_back(point);
    }

    void ToCyMatrix();

    void ToCyPara();

    void SetBadFlag();

    void CalculateLength();

    float GetRadius(){
        return mPara.at<float>(4);
    }

    float GetStart(){
        return mfstart;
    }

    float GetEnd(){
        return mfend;
    }

    cv::Mat GetPoseInverse(){
        return mTwcy.clone();
    }
    cv::Mat GetPose(){
        return mTcyw.clone();
    }

    cv::Mat GetRotation(){
        return mRcyw.clone();
    }

    cv::Mat GetTranslation(){
        return mPcyw.clone();
    }
    
    bool OnCylinder(MapPoint* pMP);
public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long unsigned int mnBALocalForKF;
    std::list<MapPoint*> mlpPastMapPoints;
    std::unordered_set<MapPoint*> mspMapPoints;
    std::list<MapPoint*> mlpCyMapPoints;
    bool bActive;               // start to estimate or help triangulate，判断当前cylinder
    bool bBuilt;                  //是否正在估计cylinder
    std::list<double> mlMPsOnAxis;     //Projections of points on the axis
    cv::Mat mParaGBA;
    double inverseVariance;
    long unsigned int mnBAGlobalForCy;
    double mMaxMPonAxis;
    double mMinMPonAxis;
  
    long unsigned int mnLoopClosingForKF;
    std::set<KeyFrame *> msCylindricalKF; 
    // std::map<KeyFrame*,size_t> msCylindricalKF;

protected:
    cv::Mat mPara;     //5 parameters, world coordinate
    cv::Mat mTwcy;     //T metrix, world coordinate
    cv::Mat mRwcy;      //Rotation matrix, world coordinate
    cv::Mat mPwcy;       //position, world coordinate
    cv::Mat mTcyw;     //T metrix, world coordinate
    cv::Mat mRcyw;      //Rotation matrix, world coordinate
    cv::Mat mPcyw;       //position, world coordinate
    cv::Mat mDir;           //direction, world coordinate

    float mfstart;
    float mfend;
    std::list<KeyFrame*> mlpCylindricalKF;
    std::map<KeyFrame*,size_t> mObservations;
    Map* mpMap;
};

} //namespace ORB_SLAM

#endif //ORB_SLAM2_MAPCYLINDER_H
