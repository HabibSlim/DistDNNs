/* Hyperbolic Tangent Activation */
#pragma once
#include "../utils.h"
#include "../types.h"
#include "layer.h"


class TanH: public Layer {
private:
    const IOMat* m_X;

public:
    TanH() : Layer("Act::TanH") {};

    void
    forward(const IOMat& X)
    {
        m_Z = X.array().tanh();
        m_X = &X;
    }

    void
    backward(const IOMat& grad_out)
    {   
        m_grad.array() = (1 - m_Z.array().square())*grad_out.array();
    }

    void update(const IOMat& grad_out) {;}
};
