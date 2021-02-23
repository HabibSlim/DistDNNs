/* Mean-Squared Error Loss */
#pragma once
#include <iostream>
#include "../types.h"
#include "loss.h"


class MSELoss : public Loss {
private:
    float
    loss_fn()
    {
        /* 
          In this case, we can re-use the computed gradient
          since m_grad for MSE is equal to pred - real.
        */
        m_loss = 2*m_grad.squaredNorm()/m_grad.rows();
        return m_loss;
    }

public:
    MSELoss() {}

    void
    error(const IOMat& pred, const IOMat& real)
    {
        m_grad_init = true;
        m_loss_computed = false;

        m_grad.noalias() = pred - real;
    }
};
