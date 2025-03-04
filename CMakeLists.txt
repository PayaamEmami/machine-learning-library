# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("NeuralNetwork" VERSION 0.1 LANGUAGES CXX)

# Set C++ standard to C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(NeuralNetworkLib
    "src/models/utils/matrix.cpp"
    "src/models/neural_network/base_network.cpp"
 "include/models/neural_network/single_layer_perceptron.h" "include/models/neural_network/multi_layer_perceptron.h" "src/models/neural_network/single_layer_perceptron.cpp" "src/models/neural_network/multi_layer_perceptron.cpp" "include/models/base_model.h" "include/models/neural_network/rnn.h" "include/models/neural_network/lstm.h" "include/models/neural_network/transformer.h" "include/models/knn/knn.h" "include/models/naive_bayes/naive_bayes.h" "include/models/svm/svm.h" "include/models/linear/linear_regression.h" "include/models/linear/logistic_regression.h" "include/models/ensemble/gradient_boosting.h" "include/models/ensemble/random_forest.h" "src/models/base_model.cpp" "src/models/knn/knn.cpp" "src/models/neural_network/rnn.cpp" "src/models/neural_network/lstm.cpp" "src/models/neural_network/transformer.cpp" "src/models/svm/svm.cpp" "src/models/linear/linear_regression.cpp" "src/models/linear/logistic_regression.cpp" "src/models/ensemble/gradient_boosting.cpp" "src/models/ensemble/random_forest.cpp" "src/models/naive_bayes/naive_bayes.cpp")

target_include_directories(NeuralNetworkLib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(NeuralNetworkDemo
    src/main.cpp
 "include/models/neural_network/single_layer_perceptron.h" "include/models/neural_network/multi_layer_perceptron.h" "src/models/neural_network/single_layer_perceptron.cpp" "src/models/neural_network/multi_layer_perceptron.cpp" "include/models/base_model.h" "include/models/neural_network/rnn.h" "include/models/neural_network/lstm.h" "include/models/neural_network/transformer.h" "include/models/knn/knn.h" "include/models/naive_bayes/naive_bayes.h" "include/models/svm/svm.h" "include/models/linear/linear_regression.h" "include/models/linear/logistic_regression.h" "include/models/ensemble/gradient_boosting.h" "include/models/ensemble/random_forest.h" "src/models/base_model.cpp" "src/models/knn/knn.cpp" "src/models/neural_network/rnn.cpp" "src/models/neural_network/lstm.cpp" "src/models/neural_network/transformer.cpp" "src/models/svm/svm.cpp" "src/models/linear/linear_regression.cpp" "src/models/linear/logistic_regression.cpp" "src/models/ensemble/gradient_boosting.cpp" "src/models/ensemble/random_forest.cpp" "src/models/naive_bayes/naive_bayes.cpp")

target_link_libraries(NeuralNetworkDemo PRIVATE NeuralNetworkLib)