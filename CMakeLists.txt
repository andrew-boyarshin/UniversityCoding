cmake_minimum_required (VERSION 3.8)

project ("UniversityCoding" LANGUAGES CXX)

find_package (doctest CONFIG REQUIRED)

#add_subdirectory ("Task1A")
#add_subdirectory ("Task2A")
#add_subdirectory ("Task3A")
add_subdirectory ("Task4B")

#[[
include(PVS-Studio.cmake)
pvs_studio_add_target(COMPILE_COMMANDS
                      FORMAT html
                      MODE GA:1,2,3 64:1,2,3 OP:1,2,3 MISRA:1,2)
]]
