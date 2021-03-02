/* Rectified Linear Unit activation layer */
#pragma once
#include "../utils.h"
#include "../types.h"
#include "layer.h"


class ReLU: public Layer {
public:
    ReLU() : Layer("Act::ReLU") {};

    void
    forward(const IOMat& X)
    {
        m_Z = X.unaryExpr([](const float x) { return float(x>0 ? x : 0); });
        m_X = &X;
    }

    void
    void backward(const IOMat& grad_out, bool no_update=false) 
    {   
        m_grad.noalias() = grad_out.cwiseProduct(
            m_X->unaryExpr([](const float x) { return float(x>0 ? 1 : 0); }));
    }

    vector<IOParam*>* serialize() { return NULL; }

    bool load(IOParam* param) { return false; };

    vector<IOParam*>* serialize_grad() { return NULL; }

    bool load_grad(IOParam* param) { return false; };
};
