# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pau/research/TDA-Timeseries

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pau/research/TDA-Timeseries

# Include any dependencies generated for this target.
include CMakeFiles/rips_persistence.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rips_persistence.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rips_persistence.dir/flags.make

CMakeFiles/rips_persistence.dir/rips_persistence.cpp.o: CMakeFiles/rips_persistence.dir/flags.make
CMakeFiles/rips_persistence.dir/rips_persistence.cpp.o: rips_persistence.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pau/research/TDA-Timeseries/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rips_persistence.dir/rips_persistence.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rips_persistence.dir/rips_persistence.cpp.o -c /home/pau/research/TDA-Timeseries/rips_persistence.cpp

CMakeFiles/rips_persistence.dir/rips_persistence.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rips_persistence.dir/rips_persistence.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pau/research/TDA-Timeseries/rips_persistence.cpp > CMakeFiles/rips_persistence.dir/rips_persistence.cpp.i

CMakeFiles/rips_persistence.dir/rips_persistence.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rips_persistence.dir/rips_persistence.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pau/research/TDA-Timeseries/rips_persistence.cpp -o CMakeFiles/rips_persistence.dir/rips_persistence.cpp.s

# Object files for target rips_persistence
rips_persistence_OBJECTS = \
"CMakeFiles/rips_persistence.dir/rips_persistence.cpp.o"

# External object files for target rips_persistence
rips_persistence_EXTERNAL_OBJECTS =

rips_persistence: CMakeFiles/rips_persistence.dir/rips_persistence.cpp.o
rips_persistence: CMakeFiles/rips_persistence.dir/build.make
rips_persistence: /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.71.0
rips_persistence: CMakeFiles/rips_persistence.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pau/research/TDA-Timeseries/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable rips_persistence"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rips_persistence.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rips_persistence.dir/build: rips_persistence

.PHONY : CMakeFiles/rips_persistence.dir/build

CMakeFiles/rips_persistence.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rips_persistence.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rips_persistence.dir/clean

CMakeFiles/rips_persistence.dir/depend:
	cd /home/pau/research/TDA-Timeseries && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pau/research/TDA-Timeseries /home/pau/research/TDA-Timeseries /home/pau/research/TDA-Timeseries /home/pau/research/TDA-Timeseries /home/pau/research/TDA-Timeseries/CMakeFiles/rips_persistence.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rips_persistence.dir/depend

