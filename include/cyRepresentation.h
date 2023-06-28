#include "Thirdparty/g2o/g2o/core/base_vertex.h"

#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_dogleg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_sba.h"


#ifndef VO_SIMULATE_CYLINDER_H
#define VO_SIMULATE_CYLINDER_H
#define pi 3.1415926535

class CylinderFittingVertex : public g2o::BaseVertex<5, Eigen::Matrix<double,5,1>> {
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW


// 重置
virtual void setToOriginImpl() override {
_estimate << 0.1, 0.1, 0.1, 0.0, 10.0;
}

// 更新
virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double,5,1> last=_estimate;
    // _estimate.block<3, 1>(0, 0) = _estimate.block<3, 1>(0, 0).normalized();
    _estimate += Eigen::Matrix<double,5,1>(update);
    // _estimate.block<3, 1>(0, 0) = _estimate.block<3, 1>(0, 0).normalized();
}

// 存盘和读盘：留空
virtual bool read(std::istream &in) {}

virtual bool write(std::ostream &out) const {}
};


class CylinderFittingEdge : public g2o::BaseBinaryEdge<1, double, g2o::VertexSBAPointXYZ, CylinderFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 计算曲线模型误差
    virtual void computeError() override {
       
        const g2o::VertexSBAPointXYZ *vPoint = static_cast<const g2o::VertexSBAPointXYZ *> (_vertices[0]);
        const CylinderFittingVertex *v = static_cast<const CylinderFittingVertex *> (_vertices[1]);
        // 前三维是李代数，第四维qx,第五维r
        const Eigen::Vector3d p = vPoint->estimate();
        const Eigen::Matrix<double,5,1> params = v->estimate();
        Eigen::Vector3d axis = params.block<3, 1>(0, 0);
        Eigen::AngleAxisd rotation_vector(axis.norm(), axis.normalized());
        Eigen::Matrix3d R;
        R = rotation_vector.toRotationMatrix();
        Eigen::Vector3d qx(params(3), 0, 0);
        double r = params(4);
        Eigen::Matrix3d A;
        A << 1, 0, 0, 0, 1, 0, 0, 0, 0;
        _error(0) = (A*(R*p + qx)).norm() - r;
       
    }

    // 计算雅可比矩阵
    virtual void linearizeOplus() override {
        
        const g2o::VertexSBAPointXYZ *vPoint = static_cast<const g2o::VertexSBAPointXYZ *> (_vertices[0]);
        const CylinderFittingVertex *v = static_cast<const CylinderFittingVertex *> (_vertices[1]);
        const Eigen::Vector3d p = vPoint->estimate();
        const Eigen::Matrix<double,5,1> params = v->estimate();
        Eigen::Vector3d axis = params.block<3, 1>(0, 0);
        Eigen::AngleAxisd rotation_vector(axis.norm(), axis.normalized());
        Eigen::Matrix3d R;
        
        R = rotation_vector.toRotationMatrix();
        Eigen::Vector3d qx(params(3, 0), 0, 0);
        double r = params(4, 0);
        Eigen::Matrix3d A;
        A << 1, 0, 0, 0, 1, 0, 0, 0, 0;
        double e = (A * (R * p + qx)).norm() - r;
        double u = e + r;
        Eigen::Vector3d Rp = R * p;
        Eigen::Matrix3d Rp_hat;
        Rp_hat << 0, -Rp(2), Rp(1), Rp(2), 0, -Rp(0), -Rp(1), Rp(0), 0;
        _jacobianOplusXj(0, 3) = pow(u, -1./2) * (2. * (A*R*p).transpose().dot(Eigen::Vector3d(1, 0, 0)) + 2. * params(3)) / 2.;
        _jacobianOplusXj.block<1, 3>(0, 0) = 1./ 2 * pow(u, -1./2) * 2. * ((A.transpose() * A * R * p).transpose() + qx.transpose()*A.transpose()*A) * (-Rp_hat);
    
        _jacobianOplusXj(0, 4) = -1.;

        _jacobianOplusXi.block<1, 3>(0, 0) = 1./ 2 * pow(u, -1./2) * 2. *(((A*R).transpose() * A * R * p).transpose() + qx.transpose()*A.transpose()*A*R);
        

    }
    double error(){
        return _error(0);
    }
    
    virtual bool read(std::istream &in) {}

    virtual bool write(std::ostream &out) const {}
private:
    
};


#endif //VO_SIMULATE_CYLINDER_H

