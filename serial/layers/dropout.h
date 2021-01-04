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
    /* Gaussian variates buffer */
    float* m_variates;

public:
    Dropout(float rate) : Layer("Net::Dropout", false, true)
    {
        if (rate <= 0 || rate >= 1)
            throw new runtime_error("[dropout.h] Invalid rate parameter.");

        m_std = sqrt(rate*(1 - rate));
        m_variates = NULL;
    };

    void
    forward(const IOMat& X)
    {
        /* Creating Gaussian mask */
        if (m_variates == NULL) {
            m_variates = random_normal(X.size(), 1, m_std);
            m_M = IOMat::Map(m_variates, X.rows(), X.cols());

            delete[] m_variates;
        } else {
            random_normal(m_M.data(), m_M.size(), 1, m_std);
        }

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
