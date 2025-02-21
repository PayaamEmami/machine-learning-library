#include "matrix.h"
#include <iostream>

namespace neural_network {

    matrix::matrix()
    {
        std::cout << "matrix constructed!\n";
    }

    matrix::~matrix()
    {
        std::cout << "matrix destroyed!\n";
    }

}
