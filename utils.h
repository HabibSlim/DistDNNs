/*
  Utility functions.
*/
#pragma once

#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <random>
#include <cmath>
#include "types.h"

using namespace std;
using namespace Eigen;


/* Dataset information */
enum
Dataset
{
    MNIST_CLASSES = 10,
    MNIST_TRAIN = 60000,
    MNIST_TEST  = 10000,

    FASHION_MNIST_TRAIN = 60000,
    FASHION_MNIST_TEST  = 10000,
};

map<Dataset,string> data_files =
        {{MNIST_TRAIN,         "./data/MNIST/train-images.ubyte"},
         {MNIST_TEST,          "./data/MNIST/test-images.ubyte"},
         {FASHION_MNIST_TRAIN, "./data/F-MNIST/train-images.ubyte"},
         {FASHION_MNIST_TEST,  "./data/F-MNIST/test-images.ubyte"}};

map<Dataset,string> label_files =
        {{MNIST_TRAIN,         "./data/MNIST/train-labels.ubyte"},
         {MNIST_TEST,          "./data/MNIST/test-labels.ubyte"},
         {FASHION_MNIST_TRAIN, "./data/F-MNIST/train-labels.ubyte"},
         {FASHION_MNIST_TEST,  "./data/F-MNIST/test-labels.ubyte"}};


/* PRNGs */
random_device RNG_DEV{};
mt19937 gen{RNG_DEV()};


int
reverse_int(int i)
{
    uchar c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/* Loading dataset images */
vector<DataMat>*
load_images(Dataset dset, int batch_size)
{
    ifstream file(data_files[dset], ios::binary);

    if(file.is_open()) {
        int magic_number = 0, image_size = 0;
        int n_rows = 0, n_cols = 0, n_images = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        if(magic_number != 2051) throw runtime_error("[load_images] Invalid image file.");

        file.read((char *)&n_images, sizeof(n_images)); n_images = reverse_int(n_images);
        file.read((char *)&n_rows,   sizeof(n_rows));   n_rows   = reverse_int(n_rows);
        file.read((char *)&n_cols,   sizeof(n_cols));   n_cols   = reverse_int(n_cols);

        image_size = n_rows*n_cols;

        /* Reading each sample data */
        uchar** _dataset = new uchar*[n_images];
        for(int i=0; i<n_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }

        /* Building data batches */
        DataMat buffer(batch_size, image_size);
        auto* _batched = new vector<DataMat>();
        int max_ims = int(n_images/batch_size)*batch_size;

        for(int i=0; i<max_ims; i+=batch_size) {
            for(int j=0; j<batch_size; j++) {
                /* Copying image data */
                for (int k=0; k<image_size; k++)
                    buffer(j, k) = _dataset[i+j][k];
            }
            _batched->push_back(buffer);
        }

        /* Freeing buffer */
        for(int i=0; i<n_images; i++)
            delete[] _dataset[i];
        delete[] _dataset;

        return _batched;
    } else {
        throw runtime_error("[load_images] Cannot open file `" + data_files[dset] + "`.");
    }
}

/* Loading dataset labels */
vector<DataMat>*
load_labels(Dataset dset, int batch_size)
{
    ifstream file(label_files[dset], ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_labels = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&n_labels, sizeof(n_labels));
        n_labels = reverse_int(n_labels);

        /* Reading each sample label */
        uchar* _dataset = new uchar[n_labels];
        for(int i=0; i<n_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }

        /* Building data batches */
        DataMat buffer(batch_size, int(MNIST_CLASSES));
        auto* _batched = new vector<DataMat>();
        int max_ims = int(n_labels/batch_size)*batch_size;

        for(int i=0; i<max_ims; i+=batch_size) {
            for(int j=0; j<batch_size; j++) {
                /* Creating one-hot vectors */
                for (int k=0; k<MNIST_CLASSES; k++)
                    buffer(j, k) = (uchar)(int(_dataset[i+j] == k));
            }
            _batched->push_back(buffer);
        }
        delete[] _dataset;

        return _batched;
    } else {
        throw runtime_error("Unable to open file `" + label_files[dset] + "`!");
    }
}

/* Generating Gaussian variates */
float*
random_normal(int n_variates, double mu, double sigma)
{
    normal_distribution<> d{mu,sigma};

    float* variates = new float[n_variates];
    for (int i=0; i<n_variates; i++)
        variates[i] = d(gen);

    return variates;
}
/* In-place alternative */
void
random_normal(float* buffer, int n_variates, double mu, double sigma)
{
    normal_distribution<> d{mu,sigma};

    for (int i=0; i<n_variates; i++)
        buffer[i] = d(gen);
}
