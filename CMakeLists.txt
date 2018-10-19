project(svd_test)

cmake_minimum_required(VERSION 3.5)

add_compile_options("-std=c++11")

find_package(Eigen3 REQUIRED)

include_directories(include/ ${EIGEN3_INCLUDE_DIRS})

enable_testing()
add_custom_target(test_verbose ${CMAKE_CTEST_COMMAND} -V)
add_subdirectory(tests)

# Mark cpp header files for installation
install(DIRECTORY include/lapack_svd/
    DESTINATION include/lapack_svd
    FILES_MATCHING PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
)