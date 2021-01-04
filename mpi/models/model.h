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

    void
    init()
    {
        if (!m_defined) {
            define();
            m_defined = true;
        }
    }

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
    }

    /* Forward pass */
    void
    forward(const IOMat& X, bool eval=false)
    {
        /* Instantiating layers */
        init();

        DAG* n = m_DAG_first;
        if (n->l == NULL)
            throw runtime_error("[model.h] No layer registered in the model.");

        /* First forward pass */
        n->l->forward(X, eval);

        while (n->next != NULL && n->next->l != NULL) {
            n = n->next;
            n->l->forward(n->prev->l->out(), eval);
        }
    }
    const IOMat&
    operator()(const IOMat& X, bool eval=false)
    {
        forward(X, eval);
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

    /* Serializing model parameters */
    vector<IOParam*>*
    serialize()
    {
        auto* params = new vector<IOParam*>();
        DAG* n = m_DAG_first;

        /* Updating with safety check */
        auto add_p = [&](Layer* l)
        {
            vector<IOParam*>* p;
            if (l->to_serialize()) {
                p = l->serialize();
                params->insert(params->end(), p->begin(), p->end());

                delete p;
            }
        };

        if (n->l == NULL)
            throw runtime_error("[model.h] No layer registered in the model.");

        add_p(n->l);
        while (n->next != NULL && n->next->l != NULL) {
            n = n->next;
            add_p(n->l);
        }

        return params;
    }

    /* Loading model parameters */
    void
    load(vector<IOParam*>* params)
    {
        DAG* n = m_DAG_first;
        bool all_loaded = false;
        unsigned int p_count = 0;

        /* Instantiating layers */
        init();

        for (auto const& p: *params) {
            /* Skipping layers with no parameters */
            while (n != NULL && (n->l != NULL && !n->l->to_serialize())) {
                n = n->next;
            }
            if (n == NULL || n->l == NULL)
                throw runtime_error("[model.h] Serialized parameters given do not match with existing layers.");

            /* Multiple parameters can belong to the same layer */
            all_loaded = n->l->load(p);

            if (all_loaded) n = n->next;
            p_count ++;
        }

        if (p_count != params->size())
            throw runtime_error("[model.h] Serialized parameters given do not match with existing layers.");
    }
};
