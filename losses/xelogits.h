/* Cross-Entropy Loss with Logits */
#pragma once
#include <iostream>
#include "../types.h"
#include "loss.h"


class XELogitsLoss : public Loss {
private:
    float
    loss_fn()
    {
        float dot_p = -m_pred->cwiseProduct(*m_real).sum();
        IOMat c_exp = m_pred->array().exp();
        IOMat c_sum = c_exp.rowwise().sum();

        return (dot_p + c_sum.array().log().sum())/(m_pred->rows());
    }

public:
    XELogitsLoss() {}

    void
    error(const IOMat& pred, const IOMat& real)
    {
        m_grad_init = true;
        m_loss_computed = false;

        /* Computing softmax matrix */
        m_grad.array() = (pred.colwise() - pred.rowwise().maxCoeff())
                         .array().exp();
        m_grad.array().colwise() /= (IOArray) m_grad.rowwise().sum();

        /* Subtracting one hot matrix */
        m_grad.array() -= real.array();

        /* Saved to compute loss */
        m_pred = &pred;
        m_real = &real;
    }
};
