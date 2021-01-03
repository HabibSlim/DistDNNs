/* Rectified Linear Unit Activation Layer*/
#pragma once
#include "../utils.h"
#include "../types.h"
#include "layer.h"


class ReLU: public Layer {
private:
    const IOMat* m_X;

public:
    ReLU() : Layer("Act::ReLU") {};

    void
    forward(const IOMat& X)
    {
        m_Z = X.unaryExpr([](const float x) { return float(x>0 ? x : 0); });
        m_X = &X;
    }

    void
    backward(const IOMat& grad_out)
    {   
        m_grad.noalias() = grad_out.cwiseProduct(
            m_X->unaryExpr([](const float x) { return float(x>0 ? 1 : 0); }));
    }

    void update(const IOMat& grad_out) {;}
};
