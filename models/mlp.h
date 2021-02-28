/* Multi Layer Perceptron */
#pragma once
#include "../layers/linear.h"
#include "../layers/leakyrelu.h"
#include "../layers/softmax.h"
#include "../layers/dropout.h"
#include "../models/model.h"


class MLP : public Model {
private:
    int m_inSize, m_outSize;
    int m_size_h1, m_size_h2;

public:
    float m_lr;

    MLP(int in_size, int out_size,
        int size_h1, int size_h2, float lr)
    {
        /* Input and output layer sizes */
        this->m_inSize  = in_size;
        this->m_outSize = out_size;

        /* Hidden layer sizes */
        this->m_size_h1 = size_h1;
        this->m_size_h2 = size_h2;

        /* Global learning rate */
        this->m_lr = lr;
    };

    void
    define()
    {
        add(new Linear(m_inSize, m_size_h1, &m_lr));
        add(new LeakyReLU());
        add(new Dropout(0.2));

        add(new Linear(m_size_h1, m_size_h2, &m_lr));
        add(new LeakyReLU());
        add(new Dropout(0.2));

        add(new Linear(m_size_h2, m_outSize, &m_lr));
        add(new Softmax());
    }

    void
    set_lr(float lr)
    {
        this->m_lr = lr;
    }
};
