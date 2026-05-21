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
};
Tensor matmul(const Tensor& A, const Tensor& B);