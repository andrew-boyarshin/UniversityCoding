# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("UniversityCoding")

find_package(doctest CONFIG REQUIRED)

# Include sub-projects.
add_subdirectory ("Task1A")
