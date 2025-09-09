# This file allows other CMake Projects to find us
# We provide general project information
# and reestablish the exported CMake Targets

# Multiple inclusion guard
if(NOT solid_dmft_FOUND)
set(solid_dmft_FOUND TRUE)
set_property(GLOBAL PROPERTY solid_dmft_FOUND TRUE)

# version
set(solid_dmft_VERSION 3.3.2 CACHE STRING "solid_dmft version")
set(solid_dmft_GIT_HASH  CACHE STRING "solid_dmft git hash")

# Root of the installation
set(solid_dmft_ROOT /local/scratch/fmartinelli/working_dir/solid_dmft.src/install CACHE STRING "solid_dmft root directory")

## Find the target dependencies
#function(find_dep)
#  get_property(${ARGV0}_FOUND GLOBAL PROPERTY ${ARGV0}_FOUND)
#  if(NOT ${ARGV0}_FOUND)
#    find_package(${ARGN} REQUIRED HINTS /local/scratch/fmartinelli/working_dir/solid_dmft.src/install)
#  endif()
#endfunction()
#find_dep(depname 1.0)

## Include the exported targets of this project
#include(/local/scratch/fmartinelli/working_dir/solid_dmft.src/install/lib/cmake/solid_dmft/solid_dmft-targets.cmake)

message(STATUS "Found solid_dmft-config.cmake with version 3.3.2, hash = , root = /local/scratch/fmartinelli/working_dir/solid_dmft.src/install")

# Was the Project built with Documentation?
set(solid_dmft_WITH_DOCUMENTATION OFF CACHE BOOL "Was solid_dmft build with documentation?")

# Was the Project built with PythonSupport?
set(solid_dmft_WITH_PYTHON_SUPPORT  CACHE BOOL "Was solid_dmft build with python support?")
if()
  set(solid_dmft_MODULE_DIR /local/scratch/fmartinelli/working_dir/solid_dmft.src/install/lib/python3.13/site-packages CACHE BOOL "The solid_dmft python module directory")
endif()

endif()
