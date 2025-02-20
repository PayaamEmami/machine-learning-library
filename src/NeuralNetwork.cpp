#include "NeuralNetwork/NeuralNetwork.h"
#include <iostream>

namespace NeuralNetwork {

    Network::Network()
    {
        std::cout << "Network constructed!\n";
    }

    Network::~Network()
    {
        std::cout << "Network destroyed!\n";
    }

}
