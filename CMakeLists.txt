cmake_minimum_required(VERSION 3.16)
project(lto)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  --debug -O3")

find_package(ifopt 2.0.1 REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(/usr/lib/python3/dist-packages/numpy/core/include)

# Include directories
include_directories(
    /usr/include/python3.8
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)




file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
# create executable
add_executable(${PROJECT_NAME} ${SOURCES})


# Set output directories dynamically
set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/build/${BUILD_FOLDER})

# Output directory for executables and build artifacts
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Define the target

# Link libraries
target_link_libraries(${PROJECT_NAME} 
  ifopt::ifopt_core
  ifopt::ifopt_ipopt
  ${Python3_LIBRARIES}
)
