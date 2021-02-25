#include <iostream>
#include "../utils.h"
#include "../models/model.h"
#include "../models/mlp.h"
#include "../losses/mse.h"
#include "../losses/xe.h"
#include "exp_utils.h"


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
    /* Loading dataset */
    int BATCH_SIZE = 64;
    vector<DataMat> *train_images, *train_labels, *test_images, *test_labels;

    /* Training set */
    train_images = load_images(MNIST_TRAIN, BATCH_SIZE);
    train_labels = load_labels(MNIST_TRAIN, BATCH_SIZE);

    /* Test set */
    test_images = load_images(MNIST_TEST, BATCH_SIZE);
    test_labels = load_labels(MNIST_TEST, BATCH_SIZE);

    /* Instantiating MLP */
    int n_features = 28*28;
    int n_labels   = 10;

    MLP net(n_features, n_labels, 256, 64, 0.01);

    /* Loss function */
    MSELoss loss;
 
    /* Training the network */
    float avg_loss = 0, val_acc;
    int n_epochs = 10;
    for (int j=0; j<n_epochs; j++) {
        /* Training for a single epoch */
        avg_loss = train_epoch_full(&net, &loss,
                                    train_images,
                                    train_labels);

        /* Computing validation accuracy */
        val_acc = evaluate(&net,
                           test_images,
                           test_labels);

        std::cout << "Epoch="     << (j+1)    << ", "
                  << "Loss="      << avg_loss << ", "
                  << "Val. Acc.=" << int(val_acc*100)/float(100) << "%"
                  << std::endl;
    }

    /* Serializing network weights */
    vector<IOParam*>* serial_net = net.serialize();

    MLP new_net(n_features, n_labels, 256, 64, 0.01);
    new_net.load(serial_net);
    val_acc = evaluate(&new_net,
                       test_images,
                       test_labels);

    std::cout << "Serialized net acc.="
              << int(val_acc*100)/float(100) << "%" << std::endl;

    /* Freeing memory */
    for (auto const& p: *serial_net) delete p;

    delete serial_net;
    delete train_images;
    delete train_labels;
    delete test_images;
    delete test_labels;

    return 0;
}
