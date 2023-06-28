#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <opencv2/core/eigen.hpp>

#include "Thirdparty/g2o/config.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/solver.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include <Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h>

#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "cyRepresentation.h"


void createLandmarks(std::vector<Eigen::Vector3d> &points)
{
    float scale = 5;
    const double k = 0.5;
    double r=10;

    points.clear();

    std::mt19937 gen{12345};
    std::uniform_real_distribution<float> theta(-3.1415926535, 3.1415926535);
    std::uniform_real_distribution<double> length{-15.0, 15.0};
    std::uniform_real_distribution<double> obstacle{0, 1};
    std::normal_distribution<double> noise{0.0, 0.5};

    Eigen::Vector3d world(0,0,1);
    Eigen::Vector3d cyAxis(1,1,1);
    Eigen::Matrix3d R_w_cy(Eigen::Quaterniond::FromTwoVectors(world, cyAxis));
    Eigen::Vector3d cyPos(1,1,2);
    Eigen::AngleAxisd rotation_axis(R_w_cy.inverse());
    std::cout << "z axis = " << R_w_cy.col(2).transpose() << std::endl;
    for (int i = 0; i < 200; i ++)
    {
        Eigen::Vector3d pt;
        float theta1=theta(gen);
        pt[0] = r*cos(theta1);
        pt[1] = r*sin(theta1);
        pt[2] = length(gen);
        pt=R_w_cy *pt+cyPos;
        points.push_back(pt);
    }

    for (int j=0; j< 0; j++) {
        Eigen::Vector3d pt;
        float theta1=0.3*theta(gen);
        pt[0] = obstacle(gen)*r*cos(theta1);
        pt[1] = obstacle(gen)*r*sin(theta1);
        pt[2] = length(gen);

        pt=R_w_cy*pt+cyPos;
        points.push_back(pt);

    }
}

std::vector<Eigen::Vector3d> addLandmarksNoise(std::vector<Eigen::Vector3d> points)
{
    size_t n=points.size();
    std::vector<Eigen::Vector3d> noisyPoints;

    std::mt19937 gen{12345};
    std::normal_distribution<double> noise{0.0, 0.025};

    // for (int i = 0; i < n; i ++) {
    //     Eigen::Vector3d pt;
    //     pt[0] = points[i][0] *(1+noise(gen)) ;
    //     pt[1] = points[i][1]  *(1+noise(gen)) ;
    //     pt[2] = points[i][2]  *(1+noise(gen)) ;
    //     noisyPoints.push_back(pt);
    // }
    for (int i = 0; i < n; i ++) {
        Eigen::Vector3d pt;
        pt[0] = points[i][0];
        pt[1] = points[i][1];
        pt[2] = points[i][2];
        noisyPoints.push_back(pt);
    }
    return noisyPoints;
}



void createCameraPose(std::vector<Eigen::Matrix4d> &v_Twc, std::vector<Eigen::Matrix4d> &v_noisyTwc)
{
    v_Twc.clear();
    v_noisyTwc.clear();

    Eigen::Vector3d world(0,0,1);
    Eigen::Vector3d cyAxis(1,1,1);
    Eigen::Matrix3d R_w_c(Eigen::Quaterniond::FromTwoVectors(world, cyAxis));
    Eigen::Vector3d cPos(20,20,20);

    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    Twc.block(0, 0, 3, 3) = R_w_c;
    v_Twc.push_back(Twc);
    v_noisyTwc.push_back(Twc);

    Twc.block(0, 0, 3, 3) = R_w_c;
    Twc.block(0, 3, 3, 1) = cPos;
    v_Twc.push_back(Twc);

    cyAxis<<0.95,1,0.95;
    R_w_c=Eigen::Quaterniond::FromTwoVectors(world, cyAxis);
    cPos<<21,23,19;

    Twc.block(0, 0, 3, 3) = R_w_c;
    Twc.block(0, 3, 3, 1) = cPos;
    v_noisyTwc.push_back(Twc);
}

void detectFeatures(const Eigen::Matrix4d &Twc, const Eigen::Matrix3d &K,
                    const std::vector<Eigen::Vector3d> &landmarks, std::vector<Eigen::Vector2i> &features, bool add_noise = true)
{
    std::mt19937 gen{12345};
    const float pixel_sigma = 1.0;
    std::normal_distribution<> d{0.0, pixel_sigma};

    Eigen::Matrix3d Rwc = Twc.block(0, 0, 3, 3);
    Eigen::Vector3d twc = Twc.block(0, 3, 3, 1);
    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = -Rcw * twc;

    features.clear();
    for (size_t l = 0; l < landmarks.size(); ++l)
    {
        Eigen::Vector3d wP = landmarks[l];
        Eigen::Vector3d cP = Rcw * wP + tcw;

        if(cP[2] < 0) continue;

        float noise_u = add_noise ? std::round(d(gen)) : 0.0f;
        float noise_v = add_noise ? std::round(d(gen)) : 0.0f;

        Eigen::Vector3d ft = K * cP;
        int u = ft[0]/ft[2] + 0.5 + noise_u;
        int v = ft[1]/ft[2] + 0.5 + noise_v;
        Eigen::Vector2i obs(u, v);
        features.push_back(obs);
//        std::cout << l << " " << obs.transpose() << std::endl;
    }

}


int main(int argc, char **argv)
{

    std::vector<Eigen::Vector3d> landmarks;
    std::vector<Eigen::Matrix4d> v_Twc;
    std::vector<Eigen::Matrix4d> v_noisyTwc;

    createLandmarks(landmarks);
    std::vector<Eigen::Vector3d> noisyLandmarks=addLandmarksNoise(landmarks);
    std::vector<Eigen::Vector2i> features_curr;
    // Setup optimizer
    g2o::SparseOptimizer optimizer;

    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // // 梯度下降方法，可以从GN, LM, DogLeg 中选
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setVerbose(true);       // 打开调试输出
    optimizer.setAlgorithm(solver);

    // 往图中增加顶点
    CylinderFittingVertex *v = new CylinderFittingVertex();
    Eigen::VectorXd abc(5);
    abc << 0.538651,0.538651,0.5,2.28696,9.63356;
    v->setEstimate(abc);
    v->setId(0);
    optimizer.addVertex(v);
//    v->setFixed(true);
    std::vector<g2o::VertexSBAPointXYZ *> vPoints;
    //    //map points
    for (size_t j = 0; j < noisyLandmarks.size(); j++) {
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(noisyLandmarks[j]);
        vPoint->setId(j + 3);
        // vPoint->setFixed(true);
        // vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        vPoints.push_back(vPoint);
    }

    std::vector<CylinderFittingEdge *>  vcy;
    // 往图中增加边
    for (size_t i = 0; i < noisyLandmarks.size(); i++) {
        CylinderFittingEdge *edge2 = new CylinderFittingEdge();
        edge2->setId(noisyLandmarks.size() * 2+i);
        edge2->setVertex(0, optimizer.vertices()[i+3]);             // 设置连接的顶点
        edge2->setVertex(1, optimizer.vertices()[0]);
        edge2->setInformation(Eigen::Matrix<double, 1, 1>::Identity()); // 信息矩阵：协方差矩阵之逆
        optimizer.addEdge(edge2);
        vcy.push_back(edge2);
    }
    optimizer.initializeOptimization();
    optimizer.optimize(40);
    auto param = v->estimate();
    std::cout<<v->estimate()<<std::endl;
    Eigen::Vector3d axis = param.block<3, 1>(0, 0);
    Eigen::AngleAxisd rotation_vector(axis.norm(), axis.normalized());
    
    std::cout << "estimate z axis = " << rotation_vector.matrix().inverse().col(2).transpose() << std::endl;

    return 0;
}

