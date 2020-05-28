add_subdirectory(/usr/src/gtest
				${PROJECT_BINARY_DIR}/gtest)
include(CTest)
set(TEST_BINARY ${PROJECT_NAME}_test)
add_executable(test_tree test.cpp)

target_link_libraries(test_tree
						gtest gtest_main)

add_test(test_tree ${PROJECT_SOURCE_DIR}/bin/test_tree)
