/* Layer abstract class */
#pragma once

#include <string>
#include <vector>
#include "../types.h"


class Layer {
protected:
    /* Output matrix */
    IOMat m_Z;
    /* Gradient matrix */
    IOMat m_grad;

    /* Reference to input matrix*/
    const IOMat* m_X;

    /* Layer name */
    const string m_name;
    /* Layer properties */
    bool m_serial;
    bool m_train_only;

public:
    Layer(const char* name,
          bool serialize=false,
          bool train_only=false) :
          m_name(name),
          m_serial(serialize),
          m_train_only(train_only) {};

    virtual ~Layer() {};

    /* Forward pass: updates the output matrix */
    virtual void forward(const IOMat& input) = 0;
    void
    forward(const IOMat& X, bool eval)
    {
        if(eval && train_only()) {
            m_Z = X;
        }
        else
            forward(X);
    }

    /* Fetching forward output */
    const IOMat& out() { return m_Z; }

    /* Backpropagation: compute grad and update weights  */
    virtual void backward(const IOMat& grad_out, bool no_update=false) = 0;
    const IOMat& grad() { return m_grad; }

    /* Layer name */
    const string& name() { return m_name; }

    /* Holds parameters to serialize */
    bool to_serialize() { return m_serial; }

    /* Train-only layer */
    bool train_only() { return m_train_only; }

    /* Serializing layer weights */
    virtual vector<IOParam*>* serialize() = 0;

    /* Loading layer parameters */
    virtual bool load(IOParam* param) = 0;

    /* Serializing model gradients */
    virtual vector<IOParam*>* serialize_grad() = 0;

    /* Loading and updating gradients */
    virtual bool load_grad(IOParam* param) = 0;
};
