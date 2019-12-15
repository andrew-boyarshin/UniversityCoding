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

include (CheckCXXCompilerFlag)
include (FindBoost)

set (Boost_USE_STATIC_LIBS ON)

find_package (Boost 1.71 REQUIRED)

find_package (Threads)

find_package (OpenSSL REQUIRED)

cmake_minimum_required (VERSION 2.8.11)
project (hyperscan C CXX)

set (HS_MAJOR_VERSION 5)
set (HS_MINOR_VERSION 2)
set (HS_PATCH_VERSION 1)
set (HS_VERSION ${HS_MAJOR_VERSION}.${HS_MINOR_VERSION}.${HS_PATCH_VERSION})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckCXXSymbolExists)
INCLUDE (CheckFunctionExists)
INCLUDE (CheckIncludeFiles)
INCLUDE (CheckIncludeFileCXX)
INCLUDE (CheckLibraryExists)
INCLUDE (CheckSymbolExists)
include (CMakeDependentOption)
include (GNUInstallDirs)
include (${CMAKE_MODULE_PATH}/platform.cmake)
include (${CMAKE_MODULE_PATH}/ragel.cmake)

find_package(PkgConfig QUIET)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Default build type 'Release with debug info'")
    set(CMAKE_BUILD_TYPE RELWITHDEBINFO CACHE STRING "" FORCE )
else()
    string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
    message(STATUS "Build type ${CMAKE_BUILD_TYPE}")
endif()

if(CMAKE_BUILD_TYPE MATCHES RELEASE|RELWITHDEBINFO|MINSIZEREL)
    message(STATUS "using release build")
    set(RELEASE_BUILD TRUE)
else()
    set(RELEASE_BUILD FALSE)
endif()

set(BINDIR "${PROJECT_BINARY_DIR}/bin")
set(LIBDIR "${PROJECT_BINARY_DIR}/lib")

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# First for the generic no-config case
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIBDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${LIBDIR}")
# Second, for multi-config builds (e.g. msvc)
foreach (OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string (TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${BINDIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${LIBDIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${LIBDIR}")
endforeach (OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)


if(CMAKE_GENERATOR STREQUAL Xcode)
    set(XCODE TRUE)
endif()

# older versions of cmake don't know things support isystem
if (XCODE OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
endif ()

set(CMAKE_INCLUDE_CURRENT_DIR 1)

include (${CMAKE_MODULE_PATH}/boost.cmake)

# -- make this work? set(python_ADDITIONAL_VERSIONS 2.7 2.6)
find_package(PythonInterp)
find_program(RAGEL ragel)

if(PYTHONINTERP_FOUND)
    set(PYTHON ${PYTHON_EXECUTABLE})
else()
    message(FATAL_ERROR "No python interpreter found")
endif()

if(${RAGEL} STREQUAL "RAGEL-NOTFOUND")
    message(FATAL_ERROR "Ragel state machine compiler not found")
endif()

option(OPTIMISE "Turns off compiler optimizations (on by default unless debug output enabled or coverage testing)" TRUE)

option(DEBUG_OUTPUT "Enable debug output (warning: very verbose)" FALSE)

if(DEBUG_OUTPUT)
    add_definitions(-DDEBUG)
    set(OPTIMISE FALSE)
endif(DEBUG_OUTPUT)

option(BUILD_SHARED_LIBS "Build shared libs instead of static" OFF)
option(BUILD_STATIC_AND_SHARED "Build shared libs as well as static" OFF)

if (BUILD_STATIC_AND_SHARED OR BUILD_SHARED_LIBS)
        message(STATUS "Building shared libraries")
else()
        message(STATUS "Building static libraries")
endif()

if (NOT BUILD_SHARED_LIBS)
    # build static libs
    set(BUILD_STATIC_LIBS ON)
    mark_as_advanced(BUILD_STATIC_LIBS)
endif ()

#for config
if (OPTIMISE)
    set(HS_OPTIMIZE ON)
endif()

CMAKE_DEPENDENT_OPTION(DISABLE_ASSERTS "Disable assert(); Asserts are enabled in debug builds, disabled in release builds" OFF "NOT RELEASE_BUILD" ON)

option(BUILD_AVX512 "Experimental: support avx512 in the fat runtime"
    OFF)

option(WINDOWS_ICC "Use Intel C++ Compiler on Windows, default off, requires ICC to be set in project" OFF)

# TODO: per platform config files?

CHECK_INCLUDE_FILES(unistd.h HAVE_UNISTD_H)
CHECK_INCLUDE_FILES(intrin.h HAVE_C_INTRIN_H)
CHECK_INCLUDE_FILE_CXX(intrin.h HAVE_CXX_INTRIN_H)
CHECK_INCLUDE_FILES(x86intrin.h HAVE_C_X86INTRIN_H)
CHECK_INCLUDE_FILE_CXX(x86intrin.h HAVE_CXX_X86INTRIN_H)

CHECK_FUNCTION_EXISTS(posix_memalign HAVE_POSIX_MEMALIGN)
CHECK_FUNCTION_EXISTS(_aligned_malloc HAVE__ALIGNED_MALLOC)

# these end up in the config file
CHECK_C_COMPILER_FLAG(-fvisibility=hidden HAS_C_HIDDEN)
CHECK_CXX_COMPILER_FLAG(-fvisibility=hidden HAS_CXX_HIDDEN)

# are we using libc++
CHECK_CXX_SYMBOL_EXISTS(_LIBCPP_VERSION ciso646 HAVE_LIBCPP)

if (RELEASE_BUILD)
    if (HAS_C_HIDDEN)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -fvisibility=hidden")
    endif()
    if (HAS_CXX_HIDDEN)
        set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -fvisibility=hidden")
    endif()
endif()

include (${CMAKE_MODULE_PATH}/arch.cmake)

# testing a builtin takes a little more work
CHECK_C_SOURCE_COMPILES("void *aa_test(void *x) { return __builtin_assume_aligned(x, 16);}\nint main(void) { return 0; }" HAVE_CC_BUILTIN_ASSUME_ALIGNED)
CHECK_CXX_SOURCE_COMPILES("void *aa_test(void *x) { return __builtin_assume_aligned(x, 16);}\nint main(void) { return 0; }" HAVE_CXX_BUILTIN_ASSUME_ALIGNED)
CHECK_C_SOURCE_COMPILES("int main(void) { __builtin_constant_p(0); }" HAVE__BUILTIN_CONSTANT_P)

# do substitutions
configure_file(${CMAKE_MODULE_PATH}/config.h.in ${PROJECT_BINARY_DIR}/config.h)
configure_file(src/hs_version.h.in ${PROJECT_BINARY_DIR}/hs_version.h)

if(NOT WIN32)
set(RAGEL_C_FLAGS "-Wno-unused")
endif()

set_source_files_properties(
    ${CMAKE_BINARY_DIR}/src/parser/Parser.cpp
    PROPERTIES
        COMPILE_FLAGS "${RAGEL_C_FLAGS}")

ragelmaker(src/parser/Parser.rl)

set_source_files_properties(
    ${CMAKE_BINARY_DIR}/src/parser/control_verbs.cpp
    PROPERTIES
        COMPILE_FLAGS "${RAGEL_C_FLAGS}")

ragelmaker(src/parser/control_verbs.rl)

SET(hs_HEADERS
    src/hs.h
    src/hs_common.h
    src/hs_compile.h
    src/hs_runtime.h
)
install(FILES ${hs_HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hs")

set (hs_exec_common_SRCS
    src/alloc.c
    src/scratch.c
    src/util/cpuid_flags.c
    src/util/cpuid_flags.h
    src/util/multibit.c
    )

set (hs_exec_SRCS
    ${hs_HEADERS}
    src/hs_version.h
    src/ue2common.h
    src/allocator.h
    src/crc32.c
    src/crc32.h
    src/report.h
    src/runtime.c
    src/stream_compress.c
    src/stream_compress.h
    src/stream_compress_impl.h
    src/fdr/fdr.c
    src/fdr/fdr.h
    src/fdr/fdr_internal.h
    src/fdr/fdr_confirm.h
    src/fdr/fdr_confirm_runtime.h
    src/fdr/flood_runtime.h
    src/fdr/fdr_loadval.h
    src/fdr/teddy.c
    src/fdr/teddy.h
    src/fdr/teddy_internal.h
    src/fdr/teddy_runtime_common.h
    src/hwlm/hwlm.c
    src/hwlm/hwlm.h
    src/hwlm/hwlm_internal.h
    src/hwlm/noodle_engine.c
    src/hwlm/noodle_engine.h
    src/hwlm/noodle_internal.h
    src/nfa/accel.c
    src/nfa/accel.h
    src/nfa/castle.c
    src/nfa/castle.h
    src/nfa/castle_internal.h
    src/nfa/gough.c
    src/nfa/gough_internal.h
    src/nfa/lbr.c
    src/nfa/lbr.h
    src/nfa/lbr_common_impl.h
    src/nfa/lbr_internal.h
    src/nfa/limex_accel.c
    src/nfa/limex_accel.h
    src/nfa/limex_exceptional.h
    src/nfa/limex_native.c
    src/nfa/limex_ring.h
    src/nfa/limex_64.c
    src/nfa/limex_simd128.c
    src/nfa/limex_simd256.c
    src/nfa/limex_simd384.c
    src/nfa/limex_simd512.c
    src/nfa/limex.h
    src/nfa/limex_common_impl.h
    src/nfa/limex_context.h
    src/nfa/limex_internal.h
    src/nfa/limex_runtime.h
    src/nfa/limex_runtime_impl.h
    src/nfa/limex_shuffle.h
    src/nfa/limex_state_impl.h
    src/nfa/mcclellan.c
    src/nfa/mcclellan.h
    src/nfa/mcclellan_common_impl.h
    src/nfa/mcclellan_internal.h
    src/nfa/mcsheng.c
    src/nfa/mcsheng_data.c
    src/nfa/mcsheng.h
    src/nfa/mcsheng_internal.h
    src/nfa/mpv.h
    src/nfa/mpv.c
    src/nfa/mpv_internal.h
    src/nfa/nfa_api.h
    src/nfa/nfa_api_dispatch.c
    src/nfa/nfa_internal.h
    src/nfa/nfa_rev_api.h
    src/nfa/repeat.c
    src/nfa/repeat.h
    src/nfa/repeat_internal.h
    src/nfa/sheng.c
    src/nfa/sheng.h
    src/nfa/sheng_defs.h
    src/nfa/sheng_impl.h
    src/nfa/sheng_impl4.h
    src/nfa/sheng_internal.h
    src/nfa/shufti.c
    src/nfa/shufti.h
    src/nfa/tamarama.c
    src/nfa/tamarama.h
    src/nfa/tamarama_internal.h
    src/nfa/truffle.c
    src/nfa/truffle.h
    src/nfa/vermicelli.h
    src/nfa/vermicelli_run.h
    src/nfa/vermicelli_sse.h
    src/som/som.h
    src/som/som_operation.h
    src/som/som_runtime.h
    src/som/som_runtime.c
    src/som/som_stream.c
    src/som/som_stream.h
    src/rose/block.c
    src/rose/catchup.h
    src/rose/catchup.c
    src/rose/infix.h
    src/rose/init.h
    src/rose/init.c
    src/rose/stream.c
    src/rose/stream_long_lit.h
    src/rose/stream_long_lit_hash.h
    src/rose/match.h
    src/rose/match.c
    src/rose/miracle.h
    src/rose/program_runtime.c
    src/rose/program_runtime.h
    src/rose/runtime.h
    src/rose/rose.h
    src/rose/rose_internal.h
    src/rose/rose_program.h
    src/rose/rose_types.h
    src/rose/rose_common.h
    src/rose/validate_mask.h
    src/rose/validate_shufti.h
    src/util/bitutils.h
    src/util/copybytes.h
    src/util/exhaust.h
    src/util/fatbit.h
    src/util/join.h
    src/util/masked_move.h
    src/util/multibit.h
    src/util/multibit.c
    src/util/multibit_compress.h
    src/util/multibit_internal.h
    src/util/pack_bits.h
    src/util/popcount.h
    src/util/pqueue.h
    src/util/scatter.h
    src/util/scatter_runtime.h
    src/util/simd_utils.h
    src/util/simd_utils.c
    src/util/state_compress.h
    src/util/state_compress.c
    src/util/unaligned.h
    src/util/uniform_ops.h
    src/database.c
    src/database.h
)

set (hs_exec_avx2_SRCS
    src/fdr/teddy_avx2.c
    src/util/masked_move.c
)


SET (hs_compile_SRCS
    ${hs_HEADERS}
    src/crc32.h
    src/database.h
    src/grey.cpp
    src/grey.h
    src/hs.cpp
    src/hs_internal.h
    src/hs_version.h
    src/scratch.h
    src/state.h
    src/ue2common.h
    src/compiler/asserts.cpp
    src/compiler/asserts.h
    src/compiler/compiler.cpp
    src/compiler/compiler.h
    src/compiler/error.cpp
    src/compiler/error.h
    src/compiler/expression_info.h
    src/fdr/engine_description.cpp
    src/fdr/engine_description.h
    src/fdr/fdr_compile.cpp
    src/fdr/fdr_compile.h
    src/fdr/fdr_compile_internal.h
    src/fdr/fdr_compile_util.cpp
    src/fdr/fdr_confirm_compile.cpp
    src/fdr/fdr_confirm.h
    src/fdr/fdr_engine_description.cpp
    src/fdr/fdr_engine_description.h
    src/fdr/fdr_internal.h
    src/fdr/flood_compile.cpp
    src/fdr/teddy_compile.cpp
    src/fdr/teddy_compile.h
    src/fdr/teddy_engine_description.cpp
    src/fdr/teddy_engine_description.h
    src/fdr/teddy_internal.h
    src/hwlm/hwlm_build.cpp
    src/hwlm/hwlm_build.h
    src/hwlm/hwlm_internal.h
    src/hwlm/hwlm_literal.cpp
    src/hwlm/hwlm_literal.h
    src/hwlm/noodle_build.cpp
    src/hwlm/noodle_build.h
    src/hwlm/noodle_internal.h
    src/nfa/accel.h
    src/nfa/accel_dfa_build_strat.cpp
    src/nfa/accel_dfa_build_strat.h
    src/nfa/accelcompile.cpp
    src/nfa/accelcompile.h
    src/nfa/callback.h
    src/nfa/castlecompile.cpp
    src/nfa/castlecompile.h
    src/nfa/dfa_build_strat.cpp
    src/nfa/dfa_build_strat.h
    src/nfa/dfa_min.cpp
    src/nfa/dfa_min.h
    src/nfa/goughcompile.cpp
    src/nfa/goughcompile.h
    src/nfa/goughcompile_accel.cpp
    src/nfa/goughcompile_internal.h
    src/nfa/goughcompile_reg.cpp
    src/nfa/mcclellan.h
    src/nfa/mcclellan_internal.h
    src/nfa/mcclellancompile.cpp
    src/nfa/mcclellancompile.h
    src/nfa/mcclellancompile_util.cpp
    src/nfa/mcclellancompile_util.h
    src/nfa/mcsheng_compile.cpp
    src/nfa/mcsheng_compile.h
    src/nfa/limex_compile.cpp
    src/nfa/limex_compile.h
    src/nfa/limex_accel.h
    src/nfa/limex_internal.h
    src/nfa/mpv_internal.h
    src/nfa/mpvcompile.cpp
    src/nfa/mpvcompile.h
    src/nfa/nfa_api.h
    src/nfa/nfa_api_queue.h
    src/nfa/nfa_api_util.h
    src/nfa/nfa_build_util.cpp
    src/nfa/nfa_build_util.h
    src/nfa/nfa_internal.h
    src/nfa/nfa_kind.h
    src/nfa/rdfa.cpp
    src/nfa/rdfa.h
    src/nfa/rdfa_graph.cpp
    src/nfa/rdfa_graph.h
    src/nfa/rdfa_merge.cpp
    src/nfa/rdfa_merge.h
    src/nfa/repeat_internal.h
    src/nfa/repeatcompile.cpp
    src/nfa/repeatcompile.h
    src/nfa/sheng_internal.h
    src/nfa/shengcompile.cpp
    src/nfa/shengcompile.h
    src/nfa/shufticompile.cpp
    src/nfa/shufticompile.h
    src/nfa/tamaramacompile.cpp
    src/nfa/tamaramacompile.h
    src/nfa/trufflecompile.cpp
    src/nfa/trufflecompile.h
    src/nfagraph/ng.cpp
    src/nfagraph/ng.h
    src/nfagraph/ng_anchored_acyclic.cpp
    src/nfagraph/ng_anchored_acyclic.h
    src/nfagraph/ng_anchored_dots.cpp
    src/nfagraph/ng_anchored_dots.h
    src/nfagraph/ng_asserts.cpp
    src/nfagraph/ng_asserts.h
    src/nfagraph/ng_builder.cpp
    src/nfagraph/ng_builder.h
    src/nfagraph/ng_calc_components.cpp
    src/nfagraph/ng_calc_components.h
    src/nfagraph/ng_cyclic_redundancy.cpp
    src/nfagraph/ng_cyclic_redundancy.h
    src/nfagraph/ng_depth.cpp
    src/nfagraph/ng_depth.h
    src/nfagraph/ng_dominators.cpp
    src/nfagraph/ng_dominators.h
    src/nfagraph/ng_edge_redundancy.cpp
    src/nfagraph/ng_edge_redundancy.h
    src/nfagraph/ng_equivalence.cpp
    src/nfagraph/ng_equivalence.h
    src/nfagraph/ng_execute.cpp
    src/nfagraph/ng_execute.h
    src/nfagraph/ng_expr_info.cpp
    src/nfagraph/ng_expr_info.h
    src/nfagraph/ng_extparam.cpp
    src/nfagraph/ng_extparam.h
    src/nfagraph/ng_fixed_width.cpp
    src/nfagraph/ng_fixed_width.h
    src/nfagraph/ng_fuzzy.cpp
    src/nfagraph/ng_fuzzy.h
    src/nfagraph/ng_haig.cpp
    src/nfagraph/ng_haig.h
    src/nfagraph/ng_holder.cpp
    src/nfagraph/ng_holder.h
    src/nfagraph/ng_is_equal.cpp
    src/nfagraph/ng_is_equal.h
    src/nfagraph/ng_lbr.cpp
    src/nfagraph/ng_lbr.h
    src/nfagraph/ng_literal_analysis.cpp
    src/nfagraph/ng_literal_analysis.h
    src/nfagraph/ng_literal_component.cpp
    src/nfagraph/ng_literal_component.h
    src/nfagraph/ng_literal_decorated.cpp
    src/nfagraph/ng_literal_decorated.h
    src/nfagraph/ng_mcclellan.cpp
    src/nfagraph/ng_mcclellan.h
    src/nfagraph/ng_mcclellan_internal.h
    src/nfagraph/ng_limex.cpp
    src/nfagraph/ng_limex.h
    src/nfagraph/ng_limex_accel.cpp
    src/nfagraph/ng_limex_accel.h
    src/nfagraph/ng_misc_opt.cpp
    src/nfagraph/ng_misc_opt.h
    src/nfagraph/ng_netflow.cpp
    src/nfagraph/ng_netflow.h
    src/nfagraph/ng_prefilter.cpp
    src/nfagraph/ng_prefilter.h
    src/nfagraph/ng_prune.cpp
    src/nfagraph/ng_prune.h
    src/nfagraph/ng_puff.cpp
    src/nfagraph/ng_puff.h
    src/nfagraph/ng_redundancy.cpp
    src/nfagraph/ng_redundancy.h
    src/nfagraph/ng_region.cpp
    src/nfagraph/ng_region.h
    src/nfagraph/ng_region_redundancy.cpp
    src/nfagraph/ng_region_redundancy.h
    src/nfagraph/ng_repeat.cpp
    src/nfagraph/ng_repeat.h
    src/nfagraph/ng_reports.cpp
    src/nfagraph/ng_reports.h
    src/nfagraph/ng_restructuring.cpp
    src/nfagraph/ng_restructuring.h
    src/nfagraph/ng_revacc.cpp
    src/nfagraph/ng_revacc.h
    src/nfagraph/ng_sep.cpp
    src/nfagraph/ng_sep.h
    src/nfagraph/ng_small_literal_set.cpp
    src/nfagraph/ng_small_literal_set.h
    src/nfagraph/ng_som.cpp
    src/nfagraph/ng_som.h
    src/nfagraph/ng_som_add_redundancy.cpp
    src/nfagraph/ng_som_add_redundancy.h
    src/nfagraph/ng_som_util.cpp
    src/nfagraph/ng_som_util.h
    src/nfagraph/ng_split.cpp
    src/nfagraph/ng_split.h
    src/nfagraph/ng_squash.cpp
    src/nfagraph/ng_squash.h
    src/nfagraph/ng_stop.cpp
    src/nfagraph/ng_stop.h
    src/nfagraph/ng_uncalc_components.cpp
    src/nfagraph/ng_uncalc_components.h
    src/nfagraph/ng_utf8.cpp
    src/nfagraph/ng_utf8.h
    src/nfagraph/ng_util.cpp
    src/nfagraph/ng_util.h
    src/nfagraph/ng_vacuous.cpp
    src/nfagraph/ng_vacuous.h
    src/nfagraph/ng_violet.cpp
    src/nfagraph/ng_violet.h
    src/nfagraph/ng_width.cpp
    src/nfagraph/ng_width.h
    src/parser/AsciiComponentClass.cpp
    src/parser/AsciiComponentClass.h
    src/parser/Component.cpp
    src/parser/Component.h
    src/parser/ComponentAlternation.cpp
    src/parser/ComponentAlternation.h
    src/parser/ComponentAssertion.cpp
    src/parser/ComponentAssertion.h
    src/parser/ComponentAtomicGroup.cpp
    src/parser/ComponentAtomicGroup.h
    src/parser/ComponentBackReference.cpp
    src/parser/ComponentBackReference.h
    src/parser/ComponentBoundary.cpp
    src/parser/ComponentBoundary.h
    src/parser/ComponentByte.cpp
    src/parser/ComponentByte.h
    src/parser/ComponentClass.cpp
    src/parser/ComponentClass.h
    src/parser/ComponentCondReference.cpp
    src/parser/ComponentCondReference.h
    src/parser/ComponentEUS.cpp
    src/parser/ComponentEUS.h
    src/parser/ComponentEmpty.cpp
    src/parser/ComponentEmpty.h
    src/parser/ComponentRepeat.cpp
    src/parser/ComponentRepeat.h
    src/parser/ComponentSequence.cpp
    src/parser/ComponentSequence.h
    src/parser/ComponentVisitor.cpp
    src/parser/ComponentVisitor.h
    src/parser/ComponentWordBoundary.cpp
    src/parser/ComponentWordBoundary.h
    src/parser/ConstComponentVisitor.cpp
    src/parser/ConstComponentVisitor.h
    src/parser/Parser.cpp
    src/parser/Parser.h
    src/parser/Utf8ComponentClass.cpp
    src/parser/Utf8ComponentClass.h
    src/parser/buildstate.cpp
    src/parser/buildstate.h
    src/parser/check_refs.cpp
    src/parser/check_refs.h
    src/parser/control_verbs.cpp
    src/parser/control_verbs.h
    src/parser/logical_combination.cpp
    src/parser/logical_combination.h
    src/parser/parse_error.cpp
    src/parser/parse_error.h
    src/parser/parser_util.cpp
    src/parser/position.h
    src/parser/position_info.h
    src/parser/prefilter.cpp
    src/parser/prefilter.h
    src/parser/shortcut_literal.cpp
    src/parser/shortcut_literal.h
    src/parser/ucp_table.cpp
    src/parser/ucp_table.h
    src/parser/unsupported.cpp
    src/parser/unsupported.h
    src/parser/utf8_validate.h
    src/parser/utf8_validate.cpp
    src/smallwrite/smallwrite_build.cpp
    src/smallwrite/smallwrite_build.h
    src/smallwrite/smallwrite_internal.h
    src/som/slot_manager.cpp
    src/som/slot_manager.h
    src/som/slot_manager_internal.h
    src/som/som.h
    src/som/som_operation.h
    src/rose/rose_build.h
    src/rose/rose_build_add.cpp
    src/rose/rose_build_add_internal.h
    src/rose/rose_build_add_mask.cpp
    src/rose/rose_build_anchored.cpp
    src/rose/rose_build_anchored.h
    src/rose/rose_build_bytecode.cpp
    src/rose/rose_build_castle.h
    src/rose/rose_build_castle.cpp
    src/rose/rose_build_compile.cpp
    src/rose/rose_build_convert.cpp
    src/rose/rose_build_convert.h
    src/rose/rose_build_dedupe.cpp
    src/rose/rose_build_engine_blob.cpp
    src/rose/rose_build_engine_blob.h
    src/rose/rose_build_exclusive.cpp
    src/rose/rose_build_exclusive.h
    src/rose/rose_build_groups.cpp
    src/rose/rose_build_groups.h
    src/rose/rose_build_impl.h
    src/rose/rose_build_infix.cpp
    src/rose/rose_build_infix.h
    src/rose/rose_build_instructions.cpp
    src/rose/rose_build_instructions.h
    src/rose/rose_build_lit_accel.cpp
    src/rose/rose_build_lit_accel.h
    src/rose/rose_build_long_lit.cpp
    src/rose/rose_build_long_lit.h
    src/rose/rose_build_lookaround.cpp
    src/rose/rose_build_lookaround.h
    src/rose/rose_build_matchers.cpp
    src/rose/rose_build_matchers.h
    src/rose/rose_build_merge.cpp
    src/rose/rose_build_merge.h
    src/rose/rose_build_misc.cpp
    src/rose/rose_build_misc.h
    src/rose/rose_build_program.cpp
    src/rose/rose_build_program.h
    src/rose/rose_build_resources.h
    src/rose/rose_build_role_aliasing.cpp
    src/rose/rose_build_scatter.cpp
    src/rose/rose_build_scatter.h
    src/rose/rose_build_util.h
    src/rose/rose_build_width.cpp
    src/rose/rose_build_width.h
    src/rose/rose_graph.h
    src/rose/rose_in_graph.h
    src/rose/rose_in_util.cpp
    src/rose/rose_in_util.h
    src/util/accel_scheme.h
    src/util/alloc.cpp
    src/util/alloc.h
    src/util/bitfield.h
    src/util/boundary_reports.h
    src/util/charreach.cpp
    src/util/charreach.h
    src/util/charreach_util.h
    src/util/clique.cpp
    src/util/clique.h
    src/util/compare.h
    src/util/compile_context.cpp
    src/util/compile_context.h
    src/util/compile_error.cpp
    src/util/compile_error.h
    src/util/container.h
    src/util/depth.cpp
    src/util/depth.h
    src/util/determinise.h
    src/util/dump_mask.cpp
    src/util/dump_mask.h
    src/util/fatbit_build.cpp
    src/util/fatbit_build.h
    src/util/flat_containers.h
    src/util/graph.h
    src/util/graph_range.h
    src/util/graph_small_color_map.h
    src/util/graph_undirected.h
    src/util/hash.h
    src/util/hash_dynamic_bitset.h
    src/util/insertion_ordered.h
    src/util/math.h
    src/util/multibit_build.cpp
    src/util/multibit_build.h
    src/util/noncopyable.h
    src/util/operators.h
    src/util/order_check.h
    src/util/partial_store.h
    src/util/partitioned_set.h
    src/util/popcount.h
    src/util/queue_index_factory.h
    src/util/report.h
    src/util/report_manager.cpp
    src/util/report_manager.h
    src/util/simd_utils.h
    src/util/small_vector.h
    src/util/target_info.cpp
    src/util/target_info.h
    src/util/ue2_graph.h
    src/util/ue2string.cpp
    src/util/ue2string.h
    src/util/unaligned.h
    src/util/unicode_def.h
    src/util/unicode_set.h
    src/util/uniform_ops.h
    src/util/unordered.h
    src/util/verify_types.h
)


# we group things by sublibraries, specifying shared and static and then
# choose which ones to build

set (LIB_VERSION ${HS_VERSION})
set (LIB_SOVERSION ${HS_MAJOR_VERSION})

set(hs_exec_SRCS ${hs_exec_SRCS} ${hs_exec_common_SRCS})

if (HAVE_AVX2)
    set(hs_exec_SRCS ${hs_exec_SRCS} ${hs_exec_avx2_SRCS})
endif()

add_executable (Task5
    "Task5.cpp"
    ${tbb_src}
    ${hs_exec_SRCS}
    src/hs_version.c
    src/hs_valid_platform.c
    ${hs_compile_SRCS}
)

add_dependencies(Task5 ragel_Parser)

target_link_libraries(Task5
    ${Boost_LIBRARIES}
    Crypt32.lib
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto)

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
    DBG_MACRO_NO_WARNING
    _CRT_SECURE_NO_WARNINGS
    BOOST_DETAIL_NO_CONTAINER_FWD
    BOOST_BEAST_USE_STD_STRING_VIEW
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    NOGDI
    NOSOUND
    NODRAWTEXT
    NOMB
    NOMETAFILE
    NOSCROLL
    NOTEXTMETRIC
    NOHELP
    NODEFERWINDOWPOS
    NOMCX
)

target_compile_features (Task5 PUBLIC cxx_std_20)
target_compile_options (Task5 PUBLIC $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/Zc:__cplusplus>)

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
        target_compile_options (Task5 PUBLIC $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/volatile:iso>)
    endif ()
    
    set(MSVC_WARNS /wd4101 /wd4146 /wd4172 /wd4200 /wd4244 /wd4267 /wd4307 /wd4334 /wd4805 /wd4996 /wd4800)
    target_compile_options (Task5 PUBLIC $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${MSVC_WARNS}>)
else ()
  set (_configure_date "Unknown")
endif()
set (TBB_CONFIG_DATE "${_configure_date}" CACHE STRING "First time that TBB was configured")
set (_configure_date "${TBB_CONFIG_DATE}")
configure_file (build/version_string.ver.in version_string.ver @ONLY)
