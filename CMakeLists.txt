cmake_minimum_required(VERSION 3.20)

project(mirage
  VERSION 0.1.0
  DESCRIPTION "C++ zeroth-order optimization library with detail acceleration and PyTorch-inspired API"
  LANGUAGES CXX
)

# Options
option(MIRAGE_ENABLE_NATIVE "Enable native architecture optimizations (-march=native)" ON)
option(MIRAGE_ENABLE_AVX512 "Enable AVX512 instructions (if available)" ON)
option(MIRAGE_ENABLE_AVX2 "Enable AVX2 instructions (if available)" ON)
option(MIRAGE_ENABLE_SSE4 "Enable SSE4.1 instructions (if available)" ON)
option(MIRAGE_ENABLE_NEON "Enable ARM NEON instructions (if available)" ON)

include(FetchContent)

find_path(EVE_INCLUDE_DIR eve/eve.hpp)
if(EVE_INCLUDE_DIR)
  message(STATUS "Found eve headers: ${EVE_INCLUDE_DIR}")
  set(MIRAGE_EVE_FOUND TRUE)
else()
  message(STATUS "eve not found — fetching via FetchContent")
  FetchContent_Declare(
    eve
    GIT_REPOSITORY https://github.com/jfalcou/eve.git
    GIT_TAG v2023.02.15
    EXCLUDE_FROM_ALL
  )
  set(EVE_BUILD_TEST OFF CACHE BOOL "" FORCE)
  set(EVE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(EVE_BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eve)
  set(MIRAGE_EVE_FETCHED TRUE)
endif()

find_path(CEREAL_INCLUDE_DIR cereal/cereal.hpp)
if(CEREAL_INCLUDE_DIR)
  message(STATUS "Found cereal headers: ${CEREAL_INCLUDE_DIR}")
  set(MIRAGE_CEREAL_FOUND TRUE)
else()
  message(STATUS "cereal not found — fetching via FetchContent")
  FetchContent_Declare(
    cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG v1.3.2
    EXCLUDE_FROM_ALL
  )
  set(JUST_INSTALL_CEREAL ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(cereal)
  set(MIRAGE_CEREAL_FETCHED TRUE)
endif()

find_path(EIGEN_INCLUDE_DIR Eigen/Core PATH_SUFFIXES eigen3)
if(EIGEN_INCLUDE_DIR)
  message(STATUS "Found Eigen headers: ${EIGEN_INCLUDE_DIR}")
  set(MIRAGE_EIGEN_FOUND TRUE)
else()
  message(STATUS "Eigen not found — fetching via FetchContent")
  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.1
    EXCLUDE_FROM_ALL
  )
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(Eigen)
  set(MIRAGE_EIGEN_FETCHED TRUE)
endif()

find_path(RANDOM123_INCLUDE_DIR Random123/threefry.h)
if(RANDOM123_INCLUDE_DIR)
  message(STATUS "Found Random123 headers: ${RANDOM123_INCLUDE_DIR}")
  set(MIRAGE_RANDOM123_FOUND TRUE)
else()
  message(STATUS "Random123 not found — fetching via FetchContent")
  FetchContent_Declare(
    random123
    GIT_REPOSITORY https://github.com/DEShawResearch/random123.git
    GIT_TAG v1.14.0
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(random123)
  set(MIRAGE_RANDOM123_FETCHED TRUE)
endif()

# Header-only library target
add_library(mirage INTERFACE)
add_library(mirage::mirage ALIAS mirage)

# Require C++23
target_compile_features(mirage INTERFACE cxx_std_23)

# Include directories
target_include_directories(mirage
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# --- Link / include deps ---

if(MIRAGE_EVE_FOUND)
  target_include_directories(mirage INTERFACE ${EVE_INCLUDE_DIR})
else()
  target_include_directories(mirage INTERFACE $<BUILD_INTERFACE:${eve_SOURCE_DIR}/include>)
endif()

if(MIRAGE_CEREAL_FOUND)
  target_include_directories(mirage INTERFACE ${CEREAL_INCLUDE_DIR})
else()
  target_include_directories(mirage INTERFACE $<BUILD_INTERFACE:${cereal_SOURCE_DIR}/include>)
endif()

if(MIRAGE_EIGEN_FOUND)
  target_include_directories(mirage INTERFACE ${EIGEN_INCLUDE_DIR})
else()
  target_include_directories(mirage INTERFACE $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>)
endif()

if(MIRAGE_RANDOM123_FOUND)
  target_include_directories(mirage INTERFACE ${RANDOM123_INCLUDE_DIR})
else()
  target_include_directories(mirage INTERFACE $<BUILD_INTERFACE:${random123_SOURCE_DIR}/include>)
endif()

# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Architecture optimizations
  if(MIRAGE_ENABLE_NATIVE)
    target_compile_options(mirage INTERFACE -march=native)
  else()
    # Check architecture
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
      if(MIRAGE_ENABLE_NEON)
        message(STATUS "ARM64 detected - NEON is enabled by default")
      endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
      if(MIRAGE_ENABLE_NEON)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
        if(COMPILER_SUPPORTS_NEON)
          target_compile_options(mirage INTERFACE -mfpu=neon)
          message(STATUS "ARM NEON enabled")
        endif()
      endif()
    else()
      if(MIRAGE_ENABLE_AVX512)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
        if(COMPILER_SUPPORTS_AVX512F)
          target_compile_options(mirage INTERFACE -mavx512f)
        endif()
      endif()

      if(MIRAGE_ENABLE_AVX2)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
          target_compile_options(mirage INTERFACE -mavx2 -mfma)
        endif()
      endif()

      if(MIRAGE_ENABLE_SSE4)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE41)
        if(COMPILER_SUPPORTS_SSE41)
          target_compile_options(mirage INTERFACE -msse4.1)
        endif()
      endif()
    endif()
  endif()

elseif(MSVC)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64")
    if(MIRAGE_ENABLE_NEON)
      message(STATUS "MSVC ARM64 detected - NEON is enabled by default")
    endif()
  else()
    if(MIRAGE_ENABLE_AVX512)
      target_compile_options(mirage INTERFACE /arch:AVX512)
    elseif(MIRAGE_ENABLE_AVX2)
      target_compile_options(mirage INTERFACE /arch:AVX2)
    elseif(MIRAGE_ENABLE_SSE4)
      target_compile_options(mirage INTERFACE /arch:SSE2)
    endif()
  endif()
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS mirage
  EXPORT mirageTargets
)

install(DIRECTORY include/mirage
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)

if(MIRAGE_EVE_FETCHED)
  install(DIRECTORY ${eve_SOURCE_DIR}/include/eve
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

if(MIRAGE_CEREAL_FETCHED)
  install(DIRECTORY ${cereal_SOURCE_DIR}/include/cereal
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

if(MIRAGE_EIGEN_FETCHED)
  install(DIRECTORY ${eigen_SOURCE_DIR}/Eigen
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

if(MIRAGE_RANDOM123_FETCHED)
  install(DIRECTORY ${random123_SOURCE_DIR}/include/Random123
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

# Export targets
install(EXPORT mirageTargets
  FILE mirageTargets.cmake
  NAMESPACE mirage::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mirage
)

# Package configuration
include(CMakePackageConfigHelpers)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/mirageConfig.cmake.in"
"@PACKAGE_INIT@

include(\"${CMAKE_CURRENT_LIST_DIR}/mirageTargets.cmake\")

check_required_components(mirage)
"
)

configure_package_config_file(
  ${CMAKE_CURRENT_BINARY_DIR}/mirageConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/mirageConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mirage
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/mirageConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/mirageConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/mirageConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mirage
)

# Export from build tree
export(EXPORT mirageTargets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/mirageTargets.cmake
  NAMESPACE mirage::
)

# Uninstall target
if(NOT TARGET uninstall)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
"if(NOT EXISTS \"${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt\")
  message(FATAL_ERROR \"Cannot find install manifest: ${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt\")
endif()

file(READ \"${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt\" files)
string(REGEX REPLACE \"\\n\" \";\" files \"\${files}\")

foreach(file \${files})
  message(STATUS \"Uninstalling \$ENV{DESTDIR}\${file}\")
  if(IS_SYMLINK \"\$ENV{DESTDIR}\${file}\" OR EXISTS \"\$ENV{DESTDIR}\${file}\")
    execute_process(
      COMMAND \"${CMAKE_COMMAND}\" -E remove \"\$ENV{DESTDIR}\${file}\"
      RESULT_VARIABLE rm_retval
    )
    if(NOT \"\${rm_retval}\" STREQUAL \"0\")
      message(FATAL_ERROR \"Problem when removing \$ENV{DESTDIR}\${file}\")
    endif()
  else()
    message(STATUS \"File \$ENV{DESTDIR}\${file} does not exist.\")
  endif()
endforeach()

set(mirage_dir \"${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/mirage\")
if(EXISTS \"\${mirage_dir}\")
  message(STATUS \"Removing mirage directory: \${mirage_dir}\")
  file(REMOVE_RECURSE \"\${mirage_dir}\")
endif()
")
  add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  )
endif()