/* Layer abstract class */
#pragma once
#include <string>
#include "../types.h"


class Layer {
protected:
    /* Output matrix */
    IOMat m_Z;
    /* Gradient matrix */
    IOMat m_grad;

    /* Layer name */
    const std::string m_name;

public:
    Layer(const char* name) : m_name(name) {};
    virtual ~Layer() {};

    /* Forward pass: updates the output matrix */
    virtual void forward(const IOMat& input) = 0;
    void
    forward(const DataMat& X)
    {
        const IOMat& X_float = X.cast<float>();
        forward(X_float);
    }
    const IOMat& out() { return m_Z; }

    /* Overload for forward passes */
    const IOMat&
    operator()(const IOMat& X)
    {
        forward(X);
        return m_Z;
    }
    const IOMat&
    operator()(const DataMat& X)
    {
        const IOMat& X_float = X.cast<float>();
        forward(X_float);
        return m_Z;
    }

    /* Backpropagation: compute grad and update weights  */
    virtual void backward(const IOMat& grad_out) = 0;
    const IOMat& grad() { return m_grad; }

    /* Updating weights */
    virtual void update(const IOMat& grad_out) = 0;

    /* Layer name */
    const std::string& name() { return m_name; }
};
