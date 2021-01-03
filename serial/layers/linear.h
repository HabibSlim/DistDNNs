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
    /* Learning rate */
    float m_lr;

    /* Weight matrix */
    IOMat m_W;
    /* Bias vector */
    IOVec m_b;

    /* Reference to input matrix*/
    const IOMat* m_X;
    /* Initial data buffer*/
    float* m_init;

public:
    Linear(int in_features, int out_features, float learning_rate)
    : Layer("Net::Linear")
    {
        this->m_inSize  = in_features;
        this->m_outSize = out_features;
        this->m_lr = learning_rate;

        init();
    }
    ~Linear()
    {
        /* Freeing initialized variates */
        delete[] m_init;
    };

    /* Initializing weight matrix: Xavier Initialization */
    void
    init()
    {
        /* Generating variates */
        double std = sqrt(2./(m_inSize + m_outSize));
        m_init = random_variates(m_inSize*m_outSize, 0, std);

        m_W = IOMat(Eigen::MatrixXf::Map(m_init, m_inSize, m_outSize));
        m_b = IOVec::Zero(1, m_outSize);
    }

    void
    forward(const IOMat& X)
    {
        /* Z = <W*X> + b */
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
        m_W = (m_W - m_lr*dW).eval();
        m_b = (m_b - m_lr*db).eval();
    }
};
