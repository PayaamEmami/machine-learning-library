#include "base_network.h"
#include <iostream>

namespace neural_network {

    base_network::base_network()
    {
        std::cout << "base_network constructed!\n";
    }

    base_network::~base_network()
    {
        std::cout << "base_network destroyed!\n";
    }

}
