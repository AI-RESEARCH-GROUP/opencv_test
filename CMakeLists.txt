cmake_minimum_required (VERSION 3.13)
project(opencv_test)

#################################
#task import third lib)
#################################
# 本机测试：
# cd cmake-build-debug
# cmake -DCMAKE_INSTALL_PREFIX=`pwd` ..
# make
# make install

set(exp "")
string(APPEND exp "PROJECT_SOURCE_DIR = " ${PROJECT_SOURCE_DIR})
message(${exp})
set(exp "")
string(APPEND exp "CMAKE_SOURCE_DIR = " ${CMAKE_SOURCE_DIR})
message(${exp})

#################################
# define self code
#################################
FILE(READ ${PROJECT_SOURCE_DIR}/src/func.cpp TEST_READ)
message(${TEST_READ})

FILE(STRINGS ${PROJECT_SOURCE_DIR}/src/func.cpp TEST_READ2)
message(${TEST_READ2})


FILE(GLOB LIB_CPP ${PROJECT_SOURCE_DIR}/src/func.cpp)
FILE(GLOB OPENCV_TEST_RUN_CPP1 ${PROJECT_SOURCE_DIR}/src/main.cpp ${LIB_CPP})
FILE(GLOB OPENCV_TEST_RUN_CPP2 ${PROJECT_SOURCE_DIR}/src/main2.cpp ${LIB_CPP})
FILE(GLOB OPENCV_TEST_H ${PROJECT_SOURCE_DIR}/include/)

add_executable(opencv_test ${OPENCV_TEST_RUN_CPP1})
add_executable(opencv_test2 ${OPENCV_TEST_RUN_CPP2})
add_executable(opencv_test3 src/study.cpp)
add_executable(opencv_test4 src/read_write.cpp)
add_executable(opencv_test5 src/draw.cpp)

include_directories(${STATIC_LIB_H})

#################################
# import third-party library
#################################
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
set(exp "")
string(APPEND exp "CMAKE_MODULE_PATH = " ${CMAKE_MODULE_PATH})
message(${exp})

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# directory of opencv library
message(${OpenCV_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )

#link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(opencv_test ${OpenCV_LIBS})
target_link_libraries(opencv_test2 ${OpenCV_LIBS})
target_link_libraries(opencv_test3 ${OpenCV_LIBS})
target_link_libraries(opencv_test4 ${OpenCV_LIBS})
target_link_libraries(opencv_test5 ${OpenCV_LIBS})


# install it
install(TARGETS opencv_test DESTINATION opencv_test_dir/bin)
install(TARGETS opencv_test2 DESTINATION opencv_test_dir/bin)
install(TARGETS opencv_test3 DESTINATION opencv_test_dir/bin)
install(TARGETS opencv_test4 DESTINATION opencv_test_dir/bin)
install(TARGETS opencv_test5 DESTINATION opencv_test_dir/bin)