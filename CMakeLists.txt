cmake_minimum_required(VERSION 3.1)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 14)

include_directories(include/)
add_executable(unit_tests test/unit_tests.cpp)

enable_testing()
add_test(UnitTests unit_tests)