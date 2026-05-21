#include "tensor.hpp"

int main() {
    Tensor A({2, 3});
    Tensor B({3, 2});

    A.at(0, 0) = 1;
    A.at(0, 1) = 2;
    A.at(0, 2) = 3;

    A.at(1, 0) = 4;
    A.at(1, 1) = 5;
    A.at(1, 2) = 6;

    B.at(0, 0) = 7;
    B.at(0, 1) = 8;

    B.at(1, 0) = 9;
    B.at(1, 1) = 10;

    B.at(2, 0) = 11;
    B.at(2, 1) = 12;

    Tensor C = matmul(A, B);

    C.print_shape();

    std::cout << C.at(0, 0) << " " << C.at(0, 1) << std::endl;
    std::cout << C.at(1, 0) << " " << C.at(1, 1) << std::endl;
    C.print_2d();
    return 0;
}