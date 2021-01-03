/* Sigmoid Activation Layer */
#pragma once
#include "../utils.h"
#include "../types.h"
#include "layer.h"


class Sigmoid: public Layer {
private:
    const IOMat* m_X;

public:
    Sigmoid() : Layer("Act::Sigmoid") {};

    void
    forward(const IOMat& X)
    {
        m_Z.array() = 1.0/(1.0 + (-X.array()).exp());
        m_X = &X;
    }

    void
    backward(const IOMat& grad_out)
    {
        IOMat sig = 1.0/(1.0 + (-m_X->array()).exp());
        sig.array() = (sig.array()*(1. - sig.array())).eval();
        m_grad = grad_out.cwiseProduct(sig);
    }

    void update(const IOMat& grad_out) {;}
};
