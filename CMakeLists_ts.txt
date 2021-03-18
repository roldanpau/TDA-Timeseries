cmake_minimum_required(VERSION 3.1)

project(hello_gudhi_world)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

#### GUDHI requires Boost - Alpha complex requires Boost program_options thread ####
find_package(Boost 1.48.0 REQUIRED COMPONENTS program_options thread)

# BOOST ISSUE result_of vs C++11
add_definitions(-DBOOST_RESULT_OF_USE_DECLTYPE)
# BOOST ISSUE with Libraries name resolution under Windows
add_definitions(-DBOOST_ALL_NO_LIB)
# problem with Visual Studio link on Boost program_options
add_definitions( -DBOOST_ALL_DYN_LINK )
# problem on Mac with boost_system and boost_thread
add_definitions( -DBOOST_SYSTEM_NO_DEPRECATED )

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

message("GUDHI_INCLUDE_DIRS = ${GUDHI_INCLUDE_DIRS}")

#### GUDHI Alpha complex requires GUDHI >= 2.0.0 ####
find_package(GUDHI 2.0.0 REQUIRED)
message("GUDHI_VERSION = ${GUDHI_VERSION}")
message("GUDHI_INCLUDE_DIRS = ${GUDHI_INCLUDE_DIRS}")
include_directories(${GUDHI_INCLUDE_DIRS})

#add_executable (rips_persistence rips_persistence.cpp)
add_executable (TDA-Timeseries TDA-Timeseries.cpp)
#target_link_libraries(rips_persistence ${CGAL_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(TDA-Timeseries ${CGAL_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY})
if (TBB_FOUND)
	#target_link_libraries(rips_persistence ${TBB_LIBRARIES})
	target_link_libraries(TDA-Timeseries ${TBB_LIBRARIES})
endif(TBB_FOUND)

