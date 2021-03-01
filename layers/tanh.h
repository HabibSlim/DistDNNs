/* Hyperbolic Tangent Activation */
#pragma once
#include "../utils.h"
#include "../types.h"
#include "layer.h"


class TanH: public Layer {
public:
    TanH() : Layer("Act::TanH") {};

    void
    forward(const IOMat& X)
    {
        m_Z = X.array().tanh();
        m_X = &X;
    }

    void
    void backward(const IOMat& grad_out, bool no_update=false) 
    {   
        m_grad.array() = (1 - m_Z.array().square())*grad_out.array();
    }

    vector<IOParam*>* serialize() { return NULL; }

    bool load(IOParam* param) { return false; };

    vector<IOParam*>* serialize_grad() { return NULL; }

    bool load_grad(IOParam* param) { return false; };
};
