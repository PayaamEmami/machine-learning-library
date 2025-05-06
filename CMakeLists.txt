cmake_minimum_required(VERSION 3.24)

# ────────────────────────────────────────────────────────────────────────────────
# Project
# ────────────────────────────────────────────────────────────────────────────────
project(mll
        VERSION 0.1.0
        DESCRIPTION "C++ Machine‑Learning Library"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Options ───────────────────────────────────────────────────────────────────────
option(MLL_WITH_CUDA       "Enable CUDA kernels"                 OFF)
option(MLL_BUILD_TESTS     "Build unit tests"                    ON)

# ────────────────────────────────────────────────────────────────────────────────
# Dependencies
# ────────────────────────────────────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

if(MLL_BUILD_TESTS)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.0
    )
    FetchContent_MakeAvailable(catch2)
    enable_testing()
endif()

# ────────────────────────────────────────────────────────────────────────────────
# mll static library
# ────────────────────────────────────────────────────────────────────────────────
file(GLOB_RECURSE MLL_SOURCES CONFIGURE_DEPENDS
     src/*.cpp
)

add_library(mll STATIC ${MLL_SOURCES})
add_library(mll::mll ALIAS mll)

target_include_directories(mll
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${eigen_SOURCE_DIR}
)

target_compile_features(mll PUBLIC cxx_std_20)

# ────────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────────
if(MLL_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ────────────────────────────────────────────────────────────────────────────────
# Examples
# ────────────────────────────────────────────────────────────────────────────────
add_subdirectory(examples)

# ────────────────────────────────────────────────────────────────────────────────
# Install rules
# ────────────────────────────────────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS mll
        EXPORT  mllTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT mllTargets
        FILE mllTargets.cmake
        NAMESPACE mll::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mll)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mllConfigVersion.cmake
    VERSION       ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/mllConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mllConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mll)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/mllConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/mllConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mll)
