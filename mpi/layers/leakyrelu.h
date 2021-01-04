#pragma once
#include "../utils.h"
#include "../types.h"
#include "layer.h"


class LeakyReLU: public Layer {
private:
    float m_leak;

public:
    LeakyReLU(float leak=0.1) : Layer("Act::LeakyReLU")
    {
        this->m_leak = leak;
    };

    void
    forward(const IOMat& X)
    {
        m_Z = X.unaryExpr([&](const float x)
            {
                return (x>0 ? x : m_leak*x);
            });
        m_X = &X;
    }

    void
    backward(const IOMat& grad_out)
    {
        m_grad.noalias() = grad_out.cwiseProduct(
            m_X->unaryExpr([&](const float x)
            {
                return (x>0 ? 1 : m_leak);
            }));
    }

    void update(const IOMat& grad_out) {;}

    vector<IOParam*>* serialize() { return NULL; }

    bool load(IOParam* param) { return false; };
};
