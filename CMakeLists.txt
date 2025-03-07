# CMakeList.txt : Top-level CMake project file for global configuration

cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Project details
project("MLLibrary" VERSION 0.1 LANGUAGES CXX)

# Set C++ standard to C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(MLLibrary
    src/models/utils/matrix.cpp
    src/models/base_model.cpp

    # Neural Networks
    src/models/neural_network/base_network.cpp
    src/models/neural_network/single_layer_perceptron.cpp
    src/models/neural_network/multi_layer_perceptron.cpp
    src/models/neural_network/rnn.cpp
    src/models/neural_network/lstm.cpp
    src/models/neural_network/transformer.cpp

    # Classical ML Algorithms
    src/models/knn/knn.cpp
    src/models/naive_bayes/naive_bayes.cpp
    src/models/svm/svm.cpp
    src/models/linear/linear_regression.cpp
    src/models/linear/logistic_regression.cpp
    src/models/ensemble/gradient_boosting.cpp
    src/models/ensemble/random_forest.cpp
    src/models/tree/decision_tree.cpp)

# Include directories
target_include_directories(MLLibrary
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Define the example executable
add_executable(MLLibraryDemo
    examples/main.cpp

    # Neural Networks
    src/models/neural_network/single_layer_perceptron.cpp
    src/models/neural_network/multi_layer_perceptron.cpp
    src/models/neural_network/rnn.cpp
    src/models/neural_network/lstm.cpp
    src/models/neural_network/transformer.cpp

    # Classical ML Algorithms
    src/models/base_model.cpp
    src/models/knn/knn.cpp
    src/models/naive_bayes/naive_bayes.cpp
    src/models/svm/svm.cpp
    src/models/linear/linear_regression.cpp
    src/models/linear/logistic_regression.cpp
    src/models/ensemble/gradient_boosting.cpp
    src/models/ensemble/random_forest.cpp
    src/models/tree/decision_tree.cpp)

# Link the library to the executable
target_link_libraries(MLLibraryDemo PRIVATE MLLibrary)