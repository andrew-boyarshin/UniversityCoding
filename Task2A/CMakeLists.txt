cmake_minimum_required (VERSION 3.8)

include (doctest)

add_executable (Task2A "Task2A.cpp")

target_compile_definitions (Task2A PRIVATE -DDOCTEST_CONFIG_IMPLEMENT)
target_compile_features (Task2A PUBLIC cxx_std_14)
target_link_libraries (Task2A PRIVATE doctest::doctest)
doctest_discover_tests (Task2A)
