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
float& Tensor::at(int i, int j) {
    if (shape.size() != 2) {
        throw std::runtime_error("Tensor is not 2D");
    }

    int rows = shape[0];
    int cols = shape[1];

    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Tensor index out of range");
    }

    return data[i * cols + j];
}

const float& Tensor::at(int i, int j) const {
    if (shape.size() != 2) {
        throw std::runtime_error("Tensor is not 2D");
    }

    int rows = shape[0];
    int cols = shape[1];

    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Tensor index out of range");
    }

    return data[i * cols + j];
}
float& Tensor::at(int i) {
    if (shape.size() != 1) {
        throw std::runtime_error("Tensor is not 1D");
    }

    int size = shape[0];

    if (i < 0 || i >= size) {
        throw std::out_of_range("Tensor index out of range");
    }

    return data[i];
}

const float& Tensor::at(int i) const {
    if (shape.size() != 1) {
        throw std::runtime_error("Tensor is not 1D");
    }

    int size = shape[0];

    if (i < 0 || i >= size) {
        throw std::out_of_range("Tensor index out of range");
    }

    return data[i];
}
void Tensor::print_2d() const {
    if (shape.size() != 2) {
        throw std::runtime_error("print_2d only supports 2D tensors");
    }

    int rows = shape[0];
    int cols = shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}
Tensor linear(const Tensor& X, const Tensor& W, const Tensor& b) {
    if (X.shape.size() != 2 || W.shape.size() != 2 || b.shape.size() != 1) {
        throw std::runtime_error("linear expects X and W to be 2D, b to be 1D");
    }

    int batch = X.shape[0];
    int in_features = X.shape[1];

    int w_in = W.shape[0];
    int out_features = W.shape[1];

    int bias_size = b.shape[0];

    if (in_features != w_in) {
        throw std::runtime_error("linear shape mismatch: X columns must match W rows");
    }

    if (bias_size != out_features) {
        throw std::runtime_error("linear shape mismatch: bias size must match W output size");
    }

    Tensor Y = matmul(X, W);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < out_features; j++) {
            Y.at(i, j) += b.at(j);
        }
    }

    return Y;
}
Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.shape.size() != 2 || B.shape.size() != 2) {
        throw std::runtime_error("matmul only supports 2D tensors");
    }

    int M = A.shape[0];
    int K = A.shape[1];

    int K2 = B.shape[0];
    int N = B.shape[1];

    if (K != K2) {
        throw std::runtime_error("matmul shape mismatch");
    }

    Tensor C({M, N});

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                sum += A.at(i, k) * B.at(k, j);
            }

            C.at(i, j) = sum;
        }
    }

    return C;
}