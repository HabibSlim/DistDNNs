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
struct IOParam
{
    float* p;
    int size;

    IOParam operator-(const IOParam& obj)
    {
        if (size != obj.size)
            throw new std::runtime_error("[IOParam] Subtracting parameters of different dimensions.");

        /* Performing operation */
        float* new_p = new float[size];
        for (int i=0; i<size; i++)
            new_p[i] = p[i] - obj.p[i];

        /* Creating new struct */
        IOParam p_;

        p_.p     = new_p;
        p_.size  = size;

        return p_;
    }

    float squaredNorm()
    {
        float norm = 0.;
        for (int i=0; i<size; i++)
            norm += p[i]*p[i];

        return norm;
    }
};


IOParam*
make_param(float *ptr, int size)
{
    IOParam* p_ = new IOParam();

    p_->p     = ptr;
    p_->size  = size;

    return p_;
}
