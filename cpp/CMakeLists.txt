cmake_minimum_required(VERSION 3.20)

project(gpsrx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GPSRX_BUILD_TESTS "Build C++ tests" ON)
option(GPSRX_BUILD_BENCHMARKS "Build C++ benchmarks" OFF)

add_library(gpsrx_core
    src/ca_code.cpp
    src/acquisition.cpp
    src/tracking.cpp
    src/navigation.cpp
    src/pvt.cpp
)
target_include_directories(gpsrx_core PUBLIC include)
target_compile_features(gpsrx_core PUBLIC cxx_std_20)

add_executable(gpsrx_app src/main.cpp)
target_link_libraries(gpsrx_app PRIVATE gpsrx_core)

if(GPSRX_BUILD_TESTS)
    enable_testing()
    add_executable(gpsrx_tests tests/smoke_test.cpp)
    target_link_libraries(gpsrx_tests PRIVATE gpsrx_core)
    add_test(NAME gpsrx_tests COMMAND gpsrx_tests)
endif()
