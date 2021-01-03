/* Model abstract class */
#pragma once
#include "../types.h"
#include "../layers/layer.h"
#include "../losses/loss.h"


class Model {
private:
    /* Output matrix */
    IOMat* m_out;

    /* Defining a linked-list type */
    struct DAG
    {
        Layer* l;
        DAG *next;
        DAG *prev;
    };
    DAG *m_DAG_first, *m_DAG_last;

    DAG*
    new_node()
    {
        DAG* n = new DAG();
        n->l    = NULL;
        n->next = NULL;
        n->prev = NULL;
        return n;
    }

    /* Flag for model definition */
    bool m_defined;

protected:
    /* Constructors */
    Model()
    {
        m_out = NULL;
        m_defined = false;

        m_DAG_first = new_node();
        m_DAG_last  = m_DAG_first;
    }

    ~Model()
    {
        DAG* n = m_DAG_first;
        if (n->l == NULL) return;

        while (n->next != NULL && n->next->l != NULL) {
            delete n->l;
            n = n->next;
            delete n->prev;
        }

        delete n->l;
        delete n->next;
        delete n;
    }

public:
    /* Defining the network DAG */
    virtual void define() = 0;

    /* Adding a new layer to the DAG */
    void
    add(Layer* new_l)
    {
        m_DAG_last->l = new_l;
        m_DAG_last->next = new_node();

        /* Saving backpointer */
        m_DAG_last->next->prev = m_DAG_last;

        /* Updating */
        m_DAG_last = m_DAG_last->next;

        std::clog << "Added: " << (m_DAG_last->prev->l->name()) << std::endl;
    }

    /* Forward pass */
    void
    forward(const IOMat& X)
    {
        if (!m_defined) {
            define();
            m_defined = true;
        }

        DAG* n = m_DAG_first;
        if (n->l == NULL)
            throw runtime_error("[model.h] No layer registered in the model.");

        /* First forward pass */
        n->l->forward(X);

        while (n->next != NULL && n->next->l != NULL) {
            n = n->next;
            n->l->forward(n->prev->l->out());
        }
    }
    const IOMat&
    operator()(const IOMat& X)
    {
        forward(X);
        return m_DAG_last->prev->l->out();
    }

    /* Backward pass */
    void
    backward(const IOMat& G)
    {
        DAG* n = m_DAG_last->prev;
        if (n->l == NULL)
            throw runtime_error("[model.h] No layer registered in the model.");

        /* First backward pass */
        n->l->backward(G);

        while (n->prev != NULL && n->prev->l != NULL) {
            n = n->prev;
            n->l->backward(n->next->l->grad());
        }
    }
};
