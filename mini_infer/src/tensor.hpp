#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>

class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;

    Tensor();
    Tensor(std::vector<int> shape);

    int numel() const;
    void print_shape() const;
    float& at(int i, int j);
    const float& at(int i, int j) const;
    void print_2d() const;
    float& at(int i);
    const float& at(int i) const;
};
Tensor matmul(const Tensor& A, const Tensor& B);
Tensor linear(const Tensor& X, const Tensor& W, const Tensor& b);
Tensor silu(const Tensor& X);
Tensor elem_mul(const Tensor& A, const Tensor& B);
Tensor mlp(
    const Tensor& X,
    const Tensor& W1,
    const Tensor& W2,
    const Tensor& W3
);