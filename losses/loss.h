/* Loss abstract class */
#pragma once
#include "../types.h"


class Loss {
protected:
    /* Previous pass refs */
    const IOMat *m_pred, *m_real;

    /* Cumulative loss */
    bool m_grad_init, m_loss_computed;
    float m_loss;

    virtual float loss_fn() = 0;

public:
    /* Gradient matrix */
    IOMat m_grad;

    Loss()
    {
        m_loss = 0.0;
        m_grad_init = false;
        m_loss_computed = false;
    }
    virtual ~Loss() {}

    /* Compute error gradient */
    virtual void error(const IOMat& pred, const IOMat& real) = 0;
    void operator()(const IOMat& pred, const IOMat& real)
    {
        error(pred, real);
    }

    /* Compute average loss */
    float loss()
    {
        if (m_grad_init) {
            if (m_loss_computed)
                return m_loss;
            else {
                m_loss_computed = true;
                return loss_fn();
            }
        } else {
            throw runtime_error("[loss.h] Gradient not computed: cannot compute average loss value.");
        }
    }

    /* Get the error gradient */
    IOMat& grad() { return m_grad; }
};
