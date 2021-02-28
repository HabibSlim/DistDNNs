/* Fully connected layer */
#pragma once
#include <iostream>
#include <math.h>
#include "../utils.h"
#include "../types.h"
#include "layer.h"


class Linear: public Layer {
private:
    /* Layer sizes */
    int m_inSize, m_outSize;

    /* Weight matrix */
    IOMat m_W;
    /* Bias vector */
    IOVec m_b;
    
    /* Parameters serialization */
    int m_serial_loaded;
    const int SERIAL_MAX = 2;

public:
    /* Learning rate */
    float* m_lr;

    Linear(int in_features, int out_features, float* learning_rate, bool to_init=true)
    : Layer("Net::Linear", true)
    {
        this->m_inSize  = in_features;
        this->m_outSize = out_features;

        this->m_lr = learning_rate;
        this->m_serial_loaded = 0;

        if (to_init) init();
    }

    /* Initializing weight matrix: Xavier Initialization */
    void
    init()
    {
        /* Generating variates */
        double std = sqrt(2./(m_inSize + m_outSize));
        float* variates = random_normal(m_inSize*m_outSize, 0, std);

        m_W = IOMat::Map(variates, m_inSize, m_outSize);
        m_b = IOVec::Zero(1, m_outSize);

        delete[] variates;
    }

    void
    forward(const IOMat& X)
    {
        /* Z = <X*W> + b */
        m_Z.noalias() = X*m_W;
        m_Z = m_Z.rowwise() + m_b;

        /* Saving last input reference */
        m_X = &X;
    }

    void
    backward(const IOMat& grad_out)
    {
        /* Computing gradient */
        m_grad.noalias() = grad_out*m_W.transpose();

        /* Updating weights */
        update(grad_out);
    }

    void
    update(const IOMat& grad_out)
    {
        IOMat dW, db;

        /* Computing d(W) */
        dW.noalias() = m_X->transpose()*grad_out;

        /* Computing d(b) */
        db.noalias() = grad_out.colwise().sum();

        /* Updating weights via SGD */
        m_W = (m_W - (*m_lr)*dW).eval();
        m_b = (m_b - (*m_lr)*db).eval();
    }

    vector<IOParam*>*
    serialize()
    {
        auto* p_ = new vector<IOParam*>();
        
        /* Adding weights and biases */
        p_->push_back(make_param(m_W.data(), m_W.size()));
        p_->push_back(make_param(m_b.data(), m_b.size()));

        return p_;
    }

    bool
    load(IOParam* param)
    {
        /* Copying buffers */
        if (m_serial_loaded == 0){
            if (param->size != m_W.size())
                throw runtime_error("[linear.h] Serialized parameters given do not match with existing layers.");
            m_W = IOMat::Map(param->p, m_inSize, m_outSize);
        } else {
            if (param->size != m_b.size())
                throw runtime_error("[linear.h] Serialized parameters given do not match with existing layers.");
            m_b = IOVec::Map(param->p, 1, m_outSize);
        }

        /* Updating layer state:
          -> return true if all parameters have been loaded
        */
        m_serial_loaded++;
        if (m_serial_loaded == SERIAL_MAX) {
            m_serial_loaded = 0;
            return true;
        } else return false;
    }
};
