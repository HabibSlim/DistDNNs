/*
  Utility functions for experiments.
*/
using namespace std;

#include <stdio.h>
#include <string>
#include <algorithm>
#include "../types.h"
#include "../losses/loss.h"
#include "../models/model.h"


/* Train a model for a single epoch */
float
train_epoch(Model* net,
            Loss* loss,
            uint* indexes, int n_batches,
            vector<DataMat>* train_ims,
            vector<DataMat>* train_labels)
{
    IOMat images, labels, pred;

    uint count = 0, k = 0;
    float avg_loss = 0.0;

    for (int i=0; i<n_batches; i++) {
        k = indexes[i];
        images = train_ims->at(k).cast<float>()/255.;
        labels = train_labels->at(k).cast<float>();

        /* Forward pass */
        pred = (*net)(images);

        /* Computing loss */
        loss->error(pred, labels);

        /* Backpropagation */
        net->backward(loss->grad());

        if (n_batches>25 && (i%(n_batches/25) == 0)) {
            avg_loss += loss->loss();
            count += 1;
        }
    }

    return (avg_loss/count);
}

/* Evaluate a model on the test set */
float
evaluate(Model* net,
         vector<DataMat>* test_ims,
         vector<DataMat>* test_labels)
{
    IOMat::Index r1, r2;
    IOMat images, labels, pred;

    int n_batches = test_ims->size(); 
    int acc_count = 0, total = n_batches*test_ims->at(0).rows();

    for (int i=0; i<n_batches; i++) {
        images = test_ims->at(i).cast<float>()/255.;
        labels = test_labels->at(i).cast<float>();

        /* Forward pass */
        pred = (*net)(images, true);

        /* Comparing real and predicted classes */
        for (int j=0; j<pred.rows(); j++) {
            pred.row(j).maxCoeff(&r1);
            labels.row(j).maxCoeff(&r2);
            acc_count += (r1 == r2);
        }
    }

    return 100*float(acc_count)/total;
}

/* Initializing an empty DataMat vector */
vector<DataMat>*
init_datamats(int n_mats, int n_rows, int n_cols)
{
    vector<DataMat>* set = new vector<DataMat>();
    for (int i=0; i<n_mats; i++)
        set->push_back(DataMat::Zero(n_rows, n_cols));
    return set;
}

/* Shuffling batch indices */
uint*
shuffle_indexes(uint *idx_tot, uint n_batches, int p)
{
    for (uint i=0; i<n_batches; i++)
        idx_tot[i] = i;

    /* Shuffling indexes */
    random_shuffle(&idx_tot[0], &idx_tot[n_batches]);

    return idx_tot;
}

/* Logging experiments results */
void
log_exp(char* fname, double* data, int size)
{
    FILE* fh = fopen(fname, "a");

    if(fh == NULL) {
        perror("Error writing to output.");
    } else {
        for (int i=0; i<size; i++)
            fprintf(fh, "%f ", data[i]);
        /* separator */
        fprintf(fh, "\n\n");
    }

    fclose(fh);
}

/* Command line parameters parser */
class ParamParser{
    private:
        vector<string> tokens;
    public:
        ParamParser(int &argc, char **argv)
        {
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(string(argv[i]));
        }

        int
        get_opt(const string &option) const
        {
            vector<string>::const_iterator itr;
            itr = find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                try {
                    return stoi(*itr);
                } catch(exception const& e) {
                    throw runtime_error("[get_opt] Non-numerical argument for "+option+" : ["+*itr+"].");
                }
            }
            throw runtime_error("[get_opt] No value specified after: ["+option+"].");
        }

        bool
        opt_exists(const string &option) const
        {
            return find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
};
