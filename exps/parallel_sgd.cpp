#include <iostream>
#include <chrono>
#include <stdio.h>
#include "mpi.h"
#include "../utils.h"
#include "../models/mlp.h"
#include "../losses/mse.h"
#include "exp_utils.h"


#define MASTER_RANK 0
#define EXP_NAME    "param_sgd"


int
main(int argc, char **argv)
{
    /* Initializing MPI */
    MPI::Init(argc, argv);

    int pcount, pid;
    pcount = MPI::COMM_WORLD.Get_size();
    pid    = MPI::COMM_WORLD.Get_rank();


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
    uint SBATCH_SIZE = BATCH_SIZE/pcount; // Subbatch size
    if (pid == MASTER_RANK) {
        printf("[param_avg] Number of processors: %d\n", pcount);
        printf("[param_avg] Batch size: %d\n", BATCH_SIZE);
        printf("[param_avg] Subbatch size: %d\n", SBATCH_SIZE);
        printf("[param_avg] Number of epochs: %d\n", N_EPOCHS);
    }


    /* Training parameters */
    uint N_FEATURES  = 28*28;
    uint N_LABELS    = 10;
    uint N_BATCHES   = int(MNIST_TRAIN/SBATCH_SIZE);

    /* Loading dataset */
    vector<DataMat> *train_images, *train_labels, *test_images, *test_labels;

    if (pid == MASTER_RANK) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        train_images = load_images(MNIST_TRAIN, SBATCH_SIZE);
        train_labels = load_labels(MNIST_TRAIN, SBATCH_SIZE);   

        /* Test set */
        test_images = load_images(MNIST_TEST, SBATCH_SIZE);
        test_labels = load_labels(MNIST_TEST, SBATCH_SIZE);     
    } else {
        train_images = init_datamats(N_BATCHES, SBATCH_SIZE, N_FEATURES);
        train_labels = init_datamats(N_BATCHES, SBATCH_SIZE, N_LABELS);

        test_images = NULL;
        test_labels = NULL;     
    }

    /* Broadcasting dataset */
    int image_batch_dim = N_FEATURES*SBATCH_SIZE; 
    int label_batch_dim = N_LABELS*SBATCH_SIZE;

	MPI::COMM_WORLD.Barrier();
    for (uint i=0; i<N_BATCHES; i++) {                                                                                                                                                                          
        MPI::COMM_WORLD.Bcast(train_images->at(i).data(), image_batch_dim,
                              MPI::UNSIGNED_CHAR,         MASTER_RANK);
        MPI::COMM_WORLD.Bcast(train_labels->at(i).data(), label_batch_dim,
                              MPI::UNSIGNED_CHAR,         MASTER_RANK);
    }


    /* Instantiating model  */
    MLP net(N_FEATURES, N_LABELS, 256, 64, 0.01);
    MSELoss loss;

    uint subbatch_idx;
    uint *data_idx;

    /* Initializing shared indices */
    if (pid == MASTER_RANK)
        data_idx = new uint[N_BATCHES];
    else
        data_idx = NULL;


    /* Experiment logs */
    double cmp_time[N_EPOCHS], val_accs[N_EPOCHS], train_losses[N_EPOCHS];

    /* Training loop */
    float val_acc, avg_loss = 0.;
    int loss_count = 0;
    chrono::time_point<chrono::high_resolution_clock> t0, t1;

    /* Serialized data */
    vector<IOParam*> *serial_grads;

    if (pid == MASTER_RANK) {
        printf("N_BATCHES=%d\n", N_BATCHES);
        printf("N_EPOCHS=%d\n", N_EPOCHS);
        printf("SBATCH_SIZE=%d\n", SBATCH_SIZE);
    }

    IOMat images, labels, pred;
    for (int j=0; j<N_EPOCHS; j++) {
        /* Splitting dataset */
        if (pid == MASTER_RANK) {
            t0 = chrono::high_resolution_clock::now();
            shuffle_indexes(data_idx, N_BATCHES);
        }

        uint k;
        for(k=0; ((k+1)*pcount-1)<N_BATCHES; k+=pcount) {
            /* Scattering batch indices */
            MPI::COMM_WORLD.Scatter(data_idx+k*pcount, 1, MPI::UNSIGNED,
                                    &subbatch_idx,     1, MPI::UNSIGNED, MASTER_RANK);

            /* Training for a single batch */
            images = train_images->at(subbatch_idx).cast<float>()/255.;
            labels = train_labels->at(subbatch_idx).cast<float>();

            /* Forward pass */
            pred = net(images);
            /* Computing loss */
            loss.error(pred, labels);

            /* Gradient-only backward pass (no weight update) */
            net.backward(loss.grad(), true);

            /* Serializing all gradients */
            serial_grads = net.serialize_grad();
            /* Reducing computed gradients */
            for (auto const& p: *serial_grads) {
                /* Applying sum reduction */
                MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, p->p, p->size,
                                          MPI::FLOAT,    MPI::SUM);

                /* Averaging values */
                for (int i=0; i<p->size; i++)
                    p->p[i] /= pcount;
            }

            /* Performing local weight updates on all nodes */
            net.load_grad(serial_grads);

            MPI::COMM_WORLD.Barrier();

            if (pid == MASTER_RANK && k%(N_BATCHES/25) == 0) {
                avg_loss += loss.loss();
                loss_count += 1;
            }
        }

        /* Evaluating metrics */
        if (pid == MASTER_RANK) {
            std::cout << "Epoch=" << (j+1);
            if (EVAL_ACC) {
                val_acc = evaluate(&net,
                                   test_images,
                                   test_labels);
                std::cout << ", Val.Acc.=" << int(val_acc*100)/float(100);
                std::cout << ", Loss.=" << int((avg_loss/loss_count)*1000)/float(1000);

                train_losses[j] = avg_loss/loss_count;
                if (j!=N_EPOCHS-1) val_accs[j] = val_acc;
                avg_loss = 0.;
            } else {
                t1 = chrono::high_resolution_clock::now();
                cmp_time[j] = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
            }
            std::cout << std::endl;
        }
    }


    /* Final net evaluation */
    if (pid == MASTER_RANK) {
        val_acc = evaluate(&net,
                            test_images,
                            test_labels);
        val_accs[N_EPOCHS-1] = val_acc;
        std::cout << "Final val. acc.="
                  << int(val_acc*100)/float(100)
                  << "%" << std::endl;

        /* Logging measurements */
        char fname[200];
        if (EVAL_ACC) {
            /* Master val. accuracies */
            sprintf(fname, "./logs/%s_N%d_B%d_acc.txt", EXP_NAME, pcount, BATCH_SIZE);
            log_exp(fname, val_accs, N_EPOCHS);

            /* Master losses */
            sprintf(fname, "./logs/%s_N%d_B%d_loss.txt", EXP_NAME, pcount, BATCH_SIZE);
            log_exp(fname, train_losses, N_EPOCHS);
        } else {
            /* Epoch durations */
            sprintf(fname, "./logs/%s_N%d_B%d_time.txt", EXP_NAME, pcount, BATCH_SIZE);    
            log_exp(fname, cmp_time, N_EPOCHS);        
        }
    }


    /* Freeing local objects */
    delete train_images;
    delete train_labels;

    if (pid == MASTER_RANK) {
        delete test_images;
        delete test_labels;
    }

    MPI::Finalize();

    return 0;
}
