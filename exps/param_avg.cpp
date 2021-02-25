#include <iostream>
#include <chrono>
#include <stdio.h>

#include "mpi.h"
#include "../utils.h"
#include "../models/mlp.h"
#include "../losses/mse.h"
#include "exp_utils.h"

#define MASTER_RANK 0

#define EXP_NAME    "param_avg"

#define EVAL_ACC    0
#define N_EPOCHS    5
#define BATCH_SIZE  64


int
main(int argc, char **argv)
{
    /* Training parameters */
    uint N_FEATURES = 28*28;
    uint N_LABELS   = 10;
    uint N_BATCHES  = int(MNIST_TRAIN/BATCH_SIZE);


    /* Initializing MPI */
    MPI::Init(argc, argv);

    int pcount, pid;
    pcount = MPI::COMM_WORLD.Get_size();
    pid    = MPI::COMM_WORLD.Get_rank();

    if (pid == MASTER_RANK)
        printf("[param_avg] Number of processors: %d\n", pcount);

    /* Averaging frequency */
    int avg_freq = (argc == 2) ? atoi(argv[1]) : 0;

    if (pid == MASTER_RANK) {
        if (avg_freq == 0)
            printf("[param_avg] Averaging a single time at the end of training.\n");
        else
            printf("[param_avg] Averaging every %d epochs\n", avg_freq);
    }


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

    /* Broadcasting dataset */
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
    MSELoss loss;

    float avg_loss = 0;
    int batch_per_p = int(N_BATCHES/pcount);
    uint *data_split, *data_idx;

    /* Initializing shared indices */
    if (pid == MASTER_RANK)
        data_idx = new uint[N_BATCHES];
    else
        data_idx = NULL;
    data_split = new uint[batch_per_p];


    /* Experiments logs */
    double *cmp_time, *val_accs, *train_losses;
    cmp_time     = new double[N_EPOCHS];
    val_accs     = new double[N_EPOCHS];
    train_losses = new double[N_EPOCHS];


    /* Training loop */
    float val_acc;
    chrono::time_point<chrono::high_resolution_clock> t0, t1;
    vector<IOParam*>* serial_net;

    for (int j=0; j<N_EPOCHS; j++) {
        /* Splitting dataset */
        if (pid == MASTER_RANK) {
            if (!EVAL_ACC) t0 = chrono::high_resolution_clock::now();
            shuffle_indexes(data_idx, N_BATCHES, pcount);
        }

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

        /* Reducing weights */
        if (avg_freq!=0 && j%avg_freq==0 && j!=N_EPOCHS-1){
            serial_net = net.serialize();
            for (auto const& p: *serial_net) {
                /* Applying sum reduction */
                MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, p->p, p->size,
                                          MPI::FLOAT, MPI::SUM);

                /* Averaging values */
                for (int i=0; i<p->size; i++)
                    p->p[i] /= pcount;
            }
            if (pid == MASTER_RANK)
                printf("[t=%d] Weights averaged!\n", j);
        }

        /* Evaluating metrics */
        if (pid == MASTER_RANK) {
            std::cout << "Epoch="  << (j+1)
                    << ", Loss=" << avg_loss;
            if (EVAL_ACC) {
                val_acc = evaluate(&net,
                                test_images,
                                test_labels);
                std::cout << ", Val.Acc.=" << int(val_acc*100)/float(100);

                train_losses[j] = avg_loss;
                if (j!=N_EPOCHS-1) val_accs[j] = val_acc;
            } else {
                t1 = chrono::high_resolution_clock::now();
                cmp_time[j] = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
            }
            std::cout << std::endl;
        }
    }

    /* Final weights reduction */
    serial_net = net.serialize();
    for (auto const& p: *serial_net) {
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
            sprintf(fname, "./logs/%s_N%d_k%d_B%d_acc.txt", EXP_NAME, pcount, avg_freq, BATCH_SIZE);
            log_exp(fname, val_accs, N_EPOCHS);

            /* Master losses */
            sprintf(fname, "./logs/%s_N%d_k%d_B%d_loss.txt", EXP_NAME, pcount, avg_freq, BATCH_SIZE);
            log_exp(fname, train_losses, N_EPOCHS);
        } else {
            /* Epoch durations */
            sprintf(fname, "./logs/%s_N%d_k%d_B%d_time.txt", EXP_NAME, pcount, avg_freq, BATCH_SIZE);    
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
