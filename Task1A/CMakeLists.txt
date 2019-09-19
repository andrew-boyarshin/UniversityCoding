cmake_minimum_required (VERSION 3.8)

include(doctest)

add_executable (Task1A "Task1A.cpp")

target_compile_definitions (Task1A PRIVATE -DDOCTEST_CONFIG_IMPLEMENT)
target_compile_features (Task1A PUBLIC cxx_std_14)
target_link_libraries (Task1A PRIVATE doctest::doctest)
doctest_discover_tests (Task1A)
