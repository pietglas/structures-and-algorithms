project(neural_network)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(data_strategies)
add_subdirectory(cost_strategies)

add_library(neural_network forward-network.cpp)

add_executable(main main.cpp)
target_link_libraries(main 
						PUBLIC OpenMP::OpenMP_CXX
						neural_network
	 					cost_strategies 
	 					data_strategies)


