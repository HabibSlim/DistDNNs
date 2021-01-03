/* Cross-Entropy Loss */
#pragma once
#include <iostream>
#include "../types.h"
#include "loss.h"


class XELoss : public Loss {
private:
    float
    loss_fn()
    {
        IOMat dot_p = m_pred->array().log().cwiseProduct(m_real->array());
        return (-dot_p.sum()/m_pred->rows());
    }

public:
    XELoss() {}

    void
    error(const IOMat& pred, const IOMat& real)
    {
        m_grad_init = true;
        m_loss_computed = false;

        /* Computing gradient */
        m_grad.noalias() = -real.cwiseQuotient(pred);
        m_grad = (m_grad.array().isFinite()).select(m_grad, 0);

        /* Saved to compute loss */
        m_pred = &pred;
        m_real = &real;
    }
};
