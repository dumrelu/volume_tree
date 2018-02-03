cmake_minimum_required(VERSION 3.1)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 14)

include_directories(include/)
add_executable(unit_tests test/unit_tests.cpp)
#target_compile_features(unit_tests PRIVATE cxx_lambda_init_captures)

add_executable(performance_with_sorting test/performance_test.cpp)
add_executable(performance_without_sorting test/performance_test.cpp)

target_compile_definitions(performance_without_sorting PRIVATE PPC_NO_SORT_INSERT=1)

enable_testing()
add_test(UnitTests unit_tests)