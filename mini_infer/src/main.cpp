#include "tensor.hpp"

int main() {
    Tensor X({1, 4});

    X.at(0, 0) = -2.0f;
    X.at(0, 1) = -1.0f;
    X.at(0, 2) = 0.0f;
    X.at(0, 3) = 2.0f;

    Tensor Y = silu(X);

    Y.print_shape();
    Y.print_2d();

    return 0;
}