cmake_minimum_required(VERSION 3.20)

project(agon
    VERSION 0.1.0
    DESCRIPTION "C++ zeroth-order optimization library with SIMD acceleration and PyTorch-inspired API"
    LANGUAGES CXX
)

# Options
option(AGON_BUILD_SHARED "Build shared library instead of static" OFF)
option(AGON_ENABLE_NATIVE "Enable native architecture optimizations (-march=native)" ON)
option(AGON_ENABLE_AVX512 "Enable AVX512 instructions (if available)" ON)
option(AGON_ENABLE_AVX2 "Enable AVX2 instructions (if available)" ON)
option(AGON_ENABLE_SSE4 "Enable SSE4.1 instructions (if available)" ON)

# Source files (automatically find all .cpp files in src/)
file(GLOB_RECURSE AGON_SOURCES src/*.cpp)

# Create library target
if(AGON_BUILD_SHARED)
    add_library(agon SHARED ${AGON_SOURCES})
else()
    add_library(agon STATIC ${AGON_SOURCES})
endif()

add_library(agon::agon ALIAS agon)

# Require C++20 for concepts
target_compile_features(agon PUBLIC cxx_std_20)

# Include directories
target_include_directories(agon
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Warning flags
    target_compile_options(agon PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

    # Architecture optimizations
    if(AGON_ENABLE_NATIVE)
        target_compile_options(agon PUBLIC -march=native)
    else()
        # Manual SIMD feature flags
        if(AGON_ENABLE_AVX512)
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
            if(COMPILER_SUPPORTS_AVX512F)
                target_compile_options(agon PUBLIC -mavx512f)
            endif()
        endif()

        if(AGON_ENABLE_AVX2)
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
            if(COMPILER_SUPPORTS_AVX2)
                target_compile_options(agon PUBLIC -mavx2 -mfma)
            endif()
        endif()

        if(AGON_ENABLE_SSE4)
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE41)
            if(COMPILER_SUPPORTS_SSE41)
                target_compile_options(agon PUBLIC -msse4.1)
            endif()
        endif()
    endif()

elseif(MSVC)
    target_compile_options(agon PRIVATE
        /W4
        /permissive-
    )

    if(AGON_ENABLE_AVX512)
        target_compile_options(agon PUBLIC /arch:AVX512)
    elseif(AGON_ENABLE_AVX2)
        target_compile_options(agon PUBLIC /arch:AVX2)
    elseif(AGON_ENABLE_SSE4)
        target_compile_options(agon PUBLIC /arch:SSE2)  # MSVC doesn't have SSE4.1 specific flag
    endif()
endif()

# Position independent code for shared libraries
set_target_properties(agon PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation
include(GNUInstallDirs)

install(TARGETS agon
    EXPORT agonTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/agon
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Export targets
install(EXPORT agonTargets
    FILE agonTargets.cmake
    NAMESPACE agon::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/agon
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/agonConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/agonConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/agon
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/agonConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/agonConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/agonConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/agon
)

# Export from build tree
export(EXPORT agonTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/agonTargets.cmake
    NAMESPACE agon::
)