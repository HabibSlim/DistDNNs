#include <iostream>
#include "utils.h"
#include "layers/linear.h"
#include "layers/relu.h"
#include "layers/leakyrelu.h"
#include "layers/softmax.h"
#include "layers/tanh.h"
#include "losses/mse.h"


int
main(int argc, char **argv)
{
    IOMat pred(4, 4);
    IOMat real(4, 4);

    pred << 0.01, 0.44, 0.54, 0.01,
            0.56, 0.01, 0.44, 0.01,
            0.01, 0.01, 0.44, 0.58,
            0.44, 0.57, 0.01, 0.01;

    real << 0.,   0.,   1.,   0.,
            1.,   0.,   0.,   0.,
            0.,   0.,   1.,   0.,
            0.,   1.,   0.,   0.;

    int count = 0;
    IOMat::Index r1, r2;
    for (int i=0; i<pred.rows(); i++) {
        pred.row(i).maxCoeff(&r1);
        real.row(i).maxCoeff(&r2);
        count += (r1 == r2);
    }

    std::cout << count << std::endl;

    return 0;
}
