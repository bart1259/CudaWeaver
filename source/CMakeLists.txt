cmake_minimum_required(VERSION 3.10.2)	#Specify the minimum CMake version
project(weaver)	#Specify the name of the project
find_package(CUDA REQUIRED)	#find the CUDA libraries
include_directories(${CUDA_INCLUDE_DIRS})	#Specify the CUDA include directories for the project

file(GLOB SOURCES
    ./*.h
    ./*.cpp
    ./*.cu
)

cuda_add_executable("${PROJECT_NAME}" ${SOURCES})	#create an executable from the given CUDA files

#specify any additional libraries here 
target_link_libraries("${PROJECT_NAME}")
