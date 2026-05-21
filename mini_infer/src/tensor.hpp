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
Tensor matmul(const Tensor& A, const Tensor& B); //basic matric multiplication
Tensor linear(const Tensor& X, const Tensor& W, const Tensor& b); //add Bias
Tensor silu(const Tensor& X); 
Tensor elem_mul(const Tensor& A, const Tensor& B); //element Multiplication
Tensor mlp(
    const Tensor& X,
    const Tensor& W1,
    const Tensor& W2,
    const Tensor& W3
); //wrapper around all the above functions
Tensor rmsnorm(const Tensor& X, const Tensor& weight, float eps = 1e-5f);
Tensor elem_add(const Tensor& A, const Tensor& B);