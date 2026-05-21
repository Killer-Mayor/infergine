#include "tensor.hpp"

int main() {
    Tensor X({1, 2});

    Tensor W1({2, 3});
    Tensor W3({2, 3});
    Tensor W2({3, 2});

    X.at(0, 0) = 1.0f;
    X.at(0, 1) = 2.0f;

    for (int i = 0; i < W1.numel(); i++) W1.data[i] = 0.1f * (i + 1);
    for (int i = 0; i < W3.numel(); i++) W3.data[i] = 0.2f * (i + 1);
    for (int i = 0; i < W2.numel(); i++) W2.data[i] = 0.3f * (i + 1);

    Tensor Y = mlp(X, W1, W2, W3);

    Y.print_shape();
    Y.print_2d();

    return 0;
}