/* Gaussian Dropout Regularization Layer*/
#pragma once
#include "../utils.h"
#include "../types.h"
#include "layer.h"


class Dropout: public Layer {
private:
    /* Dropout mask */
    IOMat m_M;
    /* Gaussian standard deviation */
    float m_std;

public:
    Dropout(float rate) : Layer("Net::Dropout", false, true)
    {
        if (rate <= 0 || rate >= 1)
            throw new runtime_error("[dropout.h] Invalid rate parameter.");

        m_std = sqrt(rate*(1 - rate));
    };

    void
    forward(const IOMat& X)
    {
        /* Generating Gaussian mask */
        float* m_init = random_normal(X.size(), 1, m_std);
        m_M = IOMat::Map(m_init, X.rows(), X.cols());
        delete[] m_init;

        /* Applying mask to input */
        m_Z = X.cwiseProduct(m_M);
    }

    void
    backward(const IOMat& grad_out)
    {   
        m_grad.noalias() = grad_out.cwiseProduct(m_M);
    }

    void update(const IOMat& grad_out) {;}

    vector<IOParam*>* serialize() { return NULL; }

    bool load(IOParam* param) { return false; }
};
