#include <iostream>
#include <chrono>
#include "../utils.h"
#include "../models/model.h"
#include "../models/mlp.h"
#include "../losses/mse.h"
#include "../losses/xe.h"
#include "exp_utils.h"


#define MASTER_RANK 0
#define EXP_NAME    "baseline"


/* Train a model on the whole dataset for a single epoch */
float
train_epoch_full(Model* net,
                 Loss* loss,
                 vector<DataMat>* train_images,
                 vector<DataMat>* train_labels)
{
    IOMat images, labels, pred;

    int n_batches = train_images->size(), count = 0; 
    float avg_loss = 0.0;

    for (int i=0; i<n_batches; i++) {
        images = train_images->at(i).cast<float>()/255.;
        labels = train_labels->at(i).cast<float>();

        /* Forward pass */
        pred = (*net)(images);

        /* Computing loss */
        loss->error(pred, labels);

        /* Backpropagation */
        net->backward(loss->grad());

        if (i%(n_batches/25) == 0) {
            avg_loss += loss->loss();
            count += 1;
        }
    }

    return (avg_loss/count);
}


int
main(int argc, char **argv)
{
    /* Parsing input parameters */
    int BATCH_SIZE = 64, N_EPOCHS = 1, EVAL_ACC = 0;
    ParamParser params(argc, argv);
    if (params.opt_exists("-batch_size")) {
        BATCH_SIZE = params.get_opt("-batch_size");
    }
    if (params.opt_exists("-n_epochs")) {
        N_EPOCHS = params.get_opt("-n_epochs");
    }
    if (params.opt_exists("-eval_acc")) {
        EVAL_ACC = params.get_opt("-eval_acc");
    }

    /* Printing all parameters */
    if (EVAL_ACC == 0)
        printf("[param_avg] Evaluating speedup.\n");
    else
        printf("[param_avg] Evaluating accuracy/training loss.\n");
    printf("[param_avg] Batch size: %d\n", BATCH_SIZE);
    printf("[param_avg] Number of epochs: %d\n", N_EPOCHS);


    /* Training parameters */
    uint N_FEATURES = 28*28;
    uint N_LABELS   = 10;


    /* Loading dataset */
    vector<DataMat> *train_images, *train_labels, *test_images, *test_labels;

    /* Training set */
    train_images = load_images(MNIST_TRAIN, BATCH_SIZE);
    train_labels = load_labels(MNIST_TRAIN, BATCH_SIZE);

    /* Test set */
    test_images = load_images(MNIST_TEST, BATCH_SIZE);
    test_labels = load_labels(MNIST_TEST, BATCH_SIZE);


    /* Instantiating MLP */
    MLP net(N_FEATURES, N_LABELS, 256, 64, 0.01);
    MSELoss loss;


    /* Experiments logs */
    double cmp_time[N_EPOCHS], val_accs[N_EPOCHS], train_losses[N_EPOCHS];

 
    /* Training the network */
    float avg_loss = 0, val_acc;
    chrono::time_point<chrono::high_resolution_clock> t0, t1;
    for (int j=0; j<N_EPOCHS; j++) {
        if (!EVAL_ACC) {
            t0 = chrono::high_resolution_clock::now();
        }

        /* Training for a single epoch */
        avg_loss = train_epoch_full(&net, &loss,
                                    train_images,
                                    train_labels);

        std::cout << "Epoch="  << (j+1)
                << ", Loss=" << avg_loss;
        if (EVAL_ACC) {
            val_acc = evaluate(&net,
                            test_images,
                            test_labels);
            std::cout << ", Val.Acc.=" << int(val_acc*100)/float(100);

            train_losses[j] = avg_loss;
            val_accs[j]     = val_acc;
        } else {
            t1 = chrono::high_resolution_clock::now();
            cmp_time[j] = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        }
        std::cout << std::endl;
    }


    /* Logging measurements */
    char fname[200];
    if (EVAL_ACC) {
        /* Master val. accuracies */
        sprintf(fname, "./logs/%s_B%d_acc.txt", EXP_NAME, BATCH_SIZE);
        log_exp(fname, val_accs, N_EPOCHS);

        /* Master losses */
        sprintf(fname, "./logs/%s_B%d_loss.txt", EXP_NAME, BATCH_SIZE);
        log_exp(fname, train_losses, N_EPOCHS);
    } else {
        /* Epoch durations */
        sprintf(fname, "./logs/%s_B%d_time.txt", EXP_NAME, BATCH_SIZE);    
        log_exp(fname, cmp_time, N_EPOCHS);
    }


    /* Freeing memory */
    delete train_images;
    delete train_labels;
    delete test_images;
    delete test_labels;

    return 0;
}
