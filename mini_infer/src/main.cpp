#include "tensor.hpp"

int main() {
    Tensor X({2, 3});
    Tensor W({3, 2});
    Tensor b({2});

    X.at(0, 0) = 1;
    X.at(0, 1) = 2;
    X.at(0, 2) = 3;

    X.at(1, 0) = 4;
    X.at(1, 1) = 5;
    X.at(1, 2) = 6;

    W.at(0, 0) = 7;
    W.at(0, 1) = 8;

    W.at(1, 0) = 9;
    W.at(1, 1) = 10;

    W.at(2, 0) = 11;
    W.at(2, 1) = 12;

    b.at(0) = 1;
    b.at(1) = 2;

    Tensor Y = linear(X, W, b);

    Y.print_shape();
    Y.print_2d();

    return 0;
}