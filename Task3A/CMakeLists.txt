cmake_minimum_required (VERSION 3.8)

include (doctest)

add_executable (Task3A "Task3A.cpp")

target_compile_definitions (Task3A PRIVATE -DDOCTEST_CONFIG_IMPLEMENT)
target_compile_features (Task3A PUBLIC cxx_std_14)
target_link_libraries (Task3A PRIVATE doctest::doctest)
doctest_discover_tests (Task3A)
