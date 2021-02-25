#include <iostream>
#include "../utils.h"
#include "../models/model.h"
#include "../layers/linear.h"
#include "../layers/relu.h"
#include "../layers/softmax.h"
#include "../layers/dropout.h"
#include "../losses/mse.h"


int
main(int argc, char **argv)
{
    IOMat data(4, 4);
    IOMat y_pred(4, 4);
    IOMat y_real(4, 4);

    /* Defining a network */
    class : public Model {
        void define() {
            add(new Linear(256, 128, 0.01));
            add(new ReLU());
            add(new Dropout(0.2));

            add(new Linear(64, 10, 0.01));
            add(new Softmax());
        }
    } net;

    MSELoss loss;

    /* Forward pass */
    y_pred = net(data);

    /* Computing loss */
    loss.error(y_pred, y_real);

    /* Backpropagation */
    net.backward(loss.grad());

    return 0;
}
