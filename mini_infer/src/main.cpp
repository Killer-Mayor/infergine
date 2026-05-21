#include "tensor.hpp"

int main() {
    Tensor X({1, 4});
    Tensor weight({4});

    X.at(0, 0) = 1.0f;
    X.at(0, 1) = 2.0f;
    X.at(0, 2) = 3.0f;
    X.at(0, 3) = 4.0f;

    for (int i = 0; i < 4; i++) {
        weight.at(i) = 1.0f;
    }

    Tensor Y = rmsnorm(X, weight);

    Y.print_shape();
    Y.print_2d();

    return 0;
}