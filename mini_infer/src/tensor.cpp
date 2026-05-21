#include "tensor.hpp"

Tensor::Tensor() {}

Tensor::Tensor(std::vector<int> shape) {
    this->shape = shape;

    int total = 1;
    for (int dim : shape) {
        total *= dim;
    }

    data.resize( total, 0.0f);
}

int Tensor::numel() const {
    int total = 1;

    for (int dim : shape) {
        total *= dim;
    }

    return total;
}

void Tensor::print_shape() const {
    std::cout << "Shape: [";

    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];

        if (i + 1 < shape.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]" << std::endl;
}