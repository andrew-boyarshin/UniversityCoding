cmake_minimum_required (VERSION 3.8)

file(GLOB tbb_src
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tbb/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rml/client/rml_tbb.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

if (WIN32)
    enable_language(ASM_MASM)
    
    if (NOT CMAKE_SIZEOF_VOID_P)
       message(FATAL_ERROR "'CMAKE_SIZEOF_VOID_P' is undefined. Please delete your build directory and rerun CMake again!")
    endif()

    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
      list(APPEND tbb_src src/tbb/intel64-masm/atomic_support.asm
        src/tbb/intel64-masm/itsx.asm src/tbb/intel64-masm/intel64_misc.asm)
      set(CMAKE_ASM_MASM_FLAGS "/DEM64T=1 ${CMAKE_ASM_MASM_FLAGS}")
    else()
      list(APPEND tbb_src src/tbb/ia32-masm/atomic_support.asm
        src/tbb/ia32-masm/itsx.asm src/tbb/ia32-masm/lock_byte.asm)
      # Enable SAFESEH feature for assembly (x86 builds only).
      set(CMAKE_ASM_MASM_FLAGS "/safeseh ${CMAKE_ASM_MASM_FLAGS}")
    endif()
else ()
    message(FATAL_ERROR "Windows-only build!")
endif()

include(CheckCXXCompilerFlag)
include(FindBoost)

set(Boost_USE_STATIC_LIBS ON)

find_package(Boost 1.71 REQUIRED)

add_executable (Task5
    "Task5.cpp"
    ${tbb_src}
)

target_link_libraries(Task5 ${Boost_LIBRARIES})

target_include_directories(Task5 PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_compile_definitions (Task5 PUBLIC
    __TBB_NO_IMPLICIT_LINKAGE
    __TBB_BUILD=1
    __TBB_DYNAMIC_LOAD_ENABLED=0
    __TBB_SOURCE_DIRECTLY_INCLUDED=1
)

target_compile_features (Task5 PUBLIC cxx_std_20)

if (UNIX)
  execute_process (COMMAND date "+%a, %d %b %Y %H:%M:%S %z"
                   OUTPUT_VARIABLE _configure_date
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
elseif (WIN32)
    execute_process (COMMAND cmd " /C date /T"
                     OUTPUT_VARIABLE _configure_date
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    target_compile_definitions (Task5 PUBLIC
        USE_WINTHREAD
        _WIN32_WINNT=0x0A00
        NTDDI_VERSION=0x0A000008
    )
    
    check_cxx_compiler_flag ("/volatile:iso" SUPPORTS_VOLATILE_FLAG)
    if (SUPPORTS_VOLATILE_FLAG)
        target_compile_options (Task5 PUBLIC "/volatile:iso")
    endif ()
else ()
  set (_configure_date "Unknown")
endif()
set (TBB_CONFIG_DATE "${_configure_date}" CACHE STRING "First time that TBB was configured")
set (_configure_date "${TBB_CONFIG_DATE}")
configure_file (build/version_string.ver.in version_string.ver @ONLY)
