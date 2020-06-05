add_subdirectory(/usr/src/gtest
				${PROJECT_BINARY_DIR}/gtest)
include(CTest)
set(TEST_BINARY ${PROJECT_NAME}_test)
add_executable(test_pq test.cpp)

target_link_libraries(test_pq
						gtest gtest_main)

add_test(test_pq ${PROJECT_SOURCE_DIR}/bin/test_pq)