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

    /* Layer name and type */
    const std::string m_name;
    bool m_serial;

public:
    Layer(const char* name, bool serializable=false) : m_name(name), m_serial(serializable) {};
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

    /* Layers holds parameters to serialize */
    bool to_serialize() { return m_serial; }

    /* Serializing layer weights */
    virtual vector<IOParam*>* serialize() = 0;

    /* Loading layer parameters */
    virtual bool load(IOParam* param) = 0;
};
