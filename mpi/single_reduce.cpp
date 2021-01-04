#include <algorithm>
#include <iostream>
#include "mpi.h"

#include "utils.h"
#include "models/model.h"
#include "models/mlp.h"
#include "losses/mse.h"
#include "losses/xe.h"


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

/* Shuffling batches and sharing to processors */
uint*
shuffle_indexes(uint *idx_tot, uint n_batches, int p)
{
    for (uint i=0; i<n_batches; i++)
        idx_tot[i] = i;

    /* Shuffling indexes */
    std::random_shuffle(&idx_tot[0], &idx_tot[n_batches]);

    return idx_tot;
}


#define MASTER_RANK 0

int
main(int argc, char **argv)
{
    /* Training parameters */
    uint BATCH_SIZE = 64;
    uint N_FEATURES = 28*28;
    uint N_LABELS   = 10;
    uint N_BATCHES  = int(MNIST_TRAIN/BATCH_SIZE);

    /* Initializing MPI */
    MPI::Init(argc, argv);

    int pcount, pid;
    pcount = MPI::COMM_WORLD.Get_size();
    pid    = MPI::COMM_WORLD.Get_rank();

    /* Loading dataset */
    vector<DataMat> *train_images, *train_labels, *test_images, *test_labels;

    if (pid == MASTER_RANK) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        train_images = load_images(MNIST_TRAIN, BATCH_SIZE);
        train_labels = load_labels(MNIST_TRAIN, BATCH_SIZE);   

        /* Test set */
        test_images = load_images(MNIST_TEST, BATCH_SIZE);
        test_labels = load_labels(MNIST_TEST, BATCH_SIZE);     
    } else {
        train_images = init_datamats(N_BATCHES, BATCH_SIZE, N_FEATURES);
        train_labels = init_datamats(N_BATCHES, BATCH_SIZE, N_LABELS);

        test_images = NULL;
        test_labels = NULL;     
    }

    /* Broadcasting data */
    int image_batch_dim = N_FEATURES*BATCH_SIZE; 
    int label_batch_dim = N_LABELS*BATCH_SIZE;

	MPI::COMM_WORLD.Barrier();
    for (uint i=0; i<N_BATCHES; i++) {                                                                                                                                                                          
        MPI::COMM_WORLD.Bcast(train_images->at(i).data(), image_batch_dim,
                              MPI::UNSIGNED_CHAR,         MASTER_RANK);
        MPI::COMM_WORLD.Bcast(train_labels->at(i).data(), label_batch_dim,
                              MPI::UNSIGNED_CHAR,         MASTER_RANK);
    }

    /* Instantiating model  */
    MLP net(N_FEATURES, N_LABELS, 256, BATCH_SIZE, 0.01);

    /* Loss function */
    MSELoss loss;

    /* Training the networks */
    float avg_loss = 0;
    int n_epochs = 10;

    int batch_per_p = int(N_BATCHES/pcount);
    uint *data_split, *data_idx;

    if (pid == MASTER_RANK)
        data_idx = new uint[N_BATCHES];
    else
        data_idx = NULL;
    data_split = new uint[batch_per_p];

    for (int j=0; j<n_epochs; j++) {
        /* Splitting dataset */
        if (pid == MASTER_RANK)
            shuffle_indexes(data_idx, N_BATCHES, pcount);

        /* Scattering batch indexes */
        MPI::COMM_WORLD.Scatter(data_idx,   batch_per_p, MPI::UNSIGNED,
                                data_split, batch_per_p, MPI::UNSIGNED, 0);

        /* Training for a single epoch */
        avg_loss = train_epoch(&net, &loss,
                               data_split, batch_per_p,
                               train_images,
                               train_labels);

        /* Printing training statistics */
        MPI::COMM_WORLD.Barrier();

        if (pid == MASTER_RANK) {
            std::cout << "Epoch="     << (j+1)   
                      << ", Loss="      << avg_loss
                      << std::endl;
        }
    }

    /* Serializing network weights */
    vector<IOParam*>* serial_net = net.serialize();

    float val_acc;
    if (pid == MASTER_RANK) {
        val_acc = evaluate(&net,
                           test_images,
                           test_labels);
        std::cout << "Root net acc.=" << int(val_acc*100)/float(100) << "%" << std::endl;
    }

    /* Weights reduction */
    for (auto const& p: *serial_net) {
        /* Applying sum reduction */
        if (pid == MASTER_RANK) {
            MPI::COMM_WORLD.Reduce(MPI::IN_PLACE, p->p, p->size,
                                   MPI::FLOAT, MPI::SUM, 0);

            /* Averaging values */
            for (int i=0; i<p->size; i++)
                p->p[i] /= pcount;
        } else {
            MPI::COMM_WORLD.Reduce(p->p, p->p, p->size,
                                   MPI::FLOAT, MPI::SUM, 0);
        }
    }

    /* Loading aggregated weights*/
    if (pid == MASTER_RANK) {
        MLP new_net(N_FEATURES, N_LABELS, 256, 64, 0.01);
        new_net.load(serial_net);

        val_acc = evaluate(&new_net,
                           test_images,
                           test_labels);
        std::cout << "Merged net acc.="     << int(val_acc*100)/float(100) << "%" << std::endl;
    }

    /* Freeing memory */
    for (auto const& p: *serial_net) delete p;
    delete serial_net;

    delete train_images;
    delete train_labels;

    if (pid == MASTER_RANK) {
        delete test_images;
        delete test_labels;
    }

    MPI::Finalize();

    return 0;
}
