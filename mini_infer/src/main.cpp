#include "tensor.hpp"

int main() {
    Tensor x({2, 3});

    x.print_shape();

    std::cout << "Number of elements: " << x.numel() << std::endl;

    return 0;
}