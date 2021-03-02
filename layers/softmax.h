/* Softmax activation Layer */
#pragma once
#include "../utils.h"
#include "../types.h"
#include "layer.h"


class Softmax: public Layer {
public:
    Softmax() : Layer("Act::Softmax") {};

    void
    forward(const IOMat& X)
    {
        /*
          Computing exp(x - max(x)) / sum(..)
          for better numerical stability.
        */
        m_Z.array() = (X.colwise() - X.rowwise().maxCoeff())
                       .array().exp();
        m_Z.array().colwise() /= (IOArray) m_Z.rowwise().sum();
    }

    void backward(const IOMat& grad_out, bool no_update=false) 
    {
        IOArray dot_p = m_Z.cwiseProduct(grad_out).rowwise().sum();
        m_grad.array() = m_Z.array()*(grad_out.array().colwise() - dot_p);
    }

    vector<IOParam*>* serialize() { return NULL; }

    bool load(IOParam* param) { return false; };

    vector<IOParam*>* serialize_grad() { return NULL; }

    bool load_grad(IOParam* param) { return false; };
};
