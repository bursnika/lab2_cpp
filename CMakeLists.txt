cmake_minimum_required(VERSION 3.10.0)
project(p2)

if (MSVC)
    add_compile_options(/W3)
    add_compile_options(/std:c++23preview)
    add_compile_options(/EHsc)
    add_compile_options(/utf-8)
    add_compile_options(/Zc:tlsGuards)
    add_compile_options(/Zc:threadSafeInit)
else()
    add_compile_options(-Wall)
    add_compile_options(-std=c++23)
    add_compile_options(-fexceptions)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    add_compile_options(-O3)
endif()

find_package(TBB REQUIRED)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lab2_cpp
        main.cpp
        timeit)

target_link_libraries(lab2_cpp PRIVATE TBB::tbb)