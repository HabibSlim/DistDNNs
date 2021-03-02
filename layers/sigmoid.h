/* Sigmoid activation Layer */
#pragma once
#include "../utils.h"
#include "../types.h"
#include "layer.h"


class Sigmoid: public Layer {
public:
    Sigmoid() : Layer("Act::Sigmoid") {};

    void
    forward(const IOMat& X)
    {
        m_Z.array() = 1.0/(1.0 + (-X.array()).exp());
        m_X = &X;
    }

    void
    void backward(const IOMat& grad_out, bool no_update=false) 
    {
        IOMat sig = 1.0/(1.0 + (-m_X->array()).exp());
        sig.array() = (sig.array()*(1. - sig.array())).eval();
        m_grad = grad_out.cwiseProduct(sig);
    }

    vector<IOParam*>* serialize() { return NULL; }

    bool load(IOParam* param) { return false; };

    vector<IOParam*>* serialize_grad() { return NULL; }

    bool load_grad(IOParam* param) { return false; };
};
