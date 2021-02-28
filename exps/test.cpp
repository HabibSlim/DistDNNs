#include <iostream>
#include "../utils.h"
#include "../models/model.h"
#include "../models/mlp.h"
#include "../losses/mse.h"
#include "../losses/xe.h"


/* Train a model for a single epoch */
float
train_epoch(Model* net,
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

/* Evaluate a model on the test set */
float
evaluate(Model* net,
         vector<DataMat>* test_images,
         vector<DataMat>* test_labels)
{
    IOMat::Index r1, r2;
    IOMat images, labels, pred;

    int n_batches = test_images->size(); 
    int acc_count = 0, total = n_batches*test_images->at(0).rows();

    for (int i=0; i<n_batches; i++) {
        images = test_images->at(i).cast<float>()/255.;
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
    int n_epochs = 1;
    for (int j=0; j<n_epochs; j++) {
        /* Training for a single epoch */
        avg_loss = train_epoch(&net, &loss,
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