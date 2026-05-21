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
};