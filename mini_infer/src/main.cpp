#include "tensor.hpp"

int main() {
    Tensor x({2, 3});

    x.at(0, 0) = 1.0f;
    x.at(0, 1) = 2.0f;
    x.at(0, 2) = 3.0f;

    x.at(1, 0) = 4.0f;
    x.at(1, 1) = 5.0f;
    x.at(1, 2) = 6.0f;

    x.print_shape();

    std::cout << "x(0, 1): " << x.at(0, 1) << std::endl;
    std::cout << "x(1, 2): " << x.at(1, 2) << std::endl;

    return 0;
}