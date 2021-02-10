cmake_minimum_required(VERSION 3.12)

project(anchor)

find_package(Python3 COMPONENTS Development)
string(SUBSTRING "${Python3_VERSION}" 0 1 Python3_MAJOR_VERSION)
string(SUBSTRING "${Python3_VERSION}" 2 1 Python3_MINOR_VERSION)
find_package(Boost COMPONENTS python${Python3_MAJOR_VERSION}${Python3_MINOR_VERSION} numpy${Python3_MAJOR_VERSION}${Python3_MINOR_VERSION} REQUIRED)

add_library(anchor MODULE anchor_wasserstein.cpp gromov.cpp matching.cpp util.cpp anchor_python.cpp)

target_compile_features(anchor PUBLIC cxx_std_11)
target_include_directories(anchor PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(anchor PRIVATE ${Boost_LIBRARIES} ${Python3_LIBRARIES})
set_target_properties(anchor PROPERTIES PREFIX "")
