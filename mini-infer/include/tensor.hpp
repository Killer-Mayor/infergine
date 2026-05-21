#pragma once

#include <vector>
#include <string>
#include <cassert>
#include <iostream>

class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;

    Tensor() = default;

    Tensor(const std::vector<int>& shape);

    int numel() const;

    float& operator[](int idx);
    const float& operator[](int idx) const;

    void print_shape() const;
};