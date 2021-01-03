#pragma once
#include "Eigen/Core"


typedef unsigned char uchar;

/* Matrix types */
typedef Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> DataMat;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> IOMat;

/* Vector types */
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> IOVec;
typedef Eigen::Array<float, Eigen::Dynamic, 1>  IOArray;

/* Serialized types */
typedef struct
{
    float* p;
    int size;
} IOParam;

IOParam*
make_param(float *ptr, int size)
{
    IOParam* p_ = new IOParam();

    p_->p     = ptr;
    p_->size  = size;

    return p_;
}
