# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/cmake-3.14.4-7gmvmpryqyfj5m42vt5qj5tb27tw7un6/bin/cmake

# The command to remove a file.
RM = /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/cmake-3.14.4-7gmvmpryqyfj5m42vt5qj5tb27tw7un6/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dealii/git/testing-repo/dealii_examples/step-1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dealii/git/testing-repo/dealii_examples/step-1

# Include any dependencies generated for this target.
include CMakeFiles/step-1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/step-1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/step-1.dir/flags.make

CMakeFiles/step-1.dir/step-1.cc.o: CMakeFiles/step-1.dir/flags.make
CMakeFiles/step-1.dir/step-1.cc.o: step-1.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dealii/git/testing-repo/dealii_examples/step-1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/step-1.dir/step-1.cc.o"
	/usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/mpich-3.3-j5u4l3i4w5xjawupwn4gsrb43tg6wntz/bin/mpic++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/step-1.dir/step-1.cc.o -c /home/dealii/git/testing-repo/dealii_examples/step-1/step-1.cc

CMakeFiles/step-1.dir/step-1.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/step-1.dir/step-1.cc.i"
	/usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/mpich-3.3-j5u4l3i4w5xjawupwn4gsrb43tg6wntz/bin/mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dealii/git/testing-repo/dealii_examples/step-1/step-1.cc > CMakeFiles/step-1.dir/step-1.cc.i

CMakeFiles/step-1.dir/step-1.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/step-1.dir/step-1.cc.s"
	/usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/mpich-3.3-j5u4l3i4w5xjawupwn4gsrb43tg6wntz/bin/mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dealii/git/testing-repo/dealii_examples/step-1/step-1.cc -o CMakeFiles/step-1.dir/step-1.cc.s

# Object files for target step-1
step__1_OBJECTS = \
"CMakeFiles/step-1.dir/step-1.cc.o"

# External object files for target step-1
step__1_EXTERNAL_OBJECTS =

step-1: CMakeFiles/step-1.dir/step-1.cc.o
step-1: CMakeFiles/step-1.dir/build.make
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/dealii-9.1.1-kbq6c5p67nir5zwpx5lbevwutndfivxz/lib/libdeal_II.g.so.9.1.1
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/intel-tbb-2019.4-esdsju3qnxprt5aqt3blhkc4lfzcantv/lib/libtbb_debug.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/boost-1.70.0-5wthozu6wbeedw6zqh7lc2zgal4bfyaa/lib/libboost_iostreams-mt.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/boost-1.70.0-5wthozu6wbeedw6zqh7lc2zgal4bfyaa/lib/libboost_serialization-mt.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/boost-1.70.0-5wthozu6wbeedw6zqh7lc2zgal4bfyaa/lib/libboost_system-mt.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/boost-1.70.0-5wthozu6wbeedw6zqh7lc2zgal4bfyaa/lib/libboost_thread-mt.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/boost-1.70.0-5wthozu6wbeedw6zqh7lc2zgal4bfyaa/lib/libboost_regex-mt.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/boost-1.70.0-5wthozu6wbeedw6zqh7lc2zgal4bfyaa/lib/libboost_chrono-mt.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/boost-1.70.0-5wthozu6wbeedw6zqh7lc2zgal4bfyaa/lib/libboost_date_time-mt.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/boost-1.70.0-5wthozu6wbeedw6zqh7lc2zgal4bfyaa/lib/libboost_atomic-mt.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/ginkgo-1.0.0-unhqrzcwydp6yf2yvbp5r4owbs6g7oel/lib/libginkgo.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/ginkgo-1.0.0-unhqrzcwydp6yf2yvbp5r4owbs6g7oel/lib/libginkgo_reference.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/ginkgo-1.0.0-unhqrzcwydp6yf2yvbp5r4owbs6g7oel/lib/libginkgo_omp.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/ginkgo-1.0.0-unhqrzcwydp6yf2yvbp5r4owbs6g7oel/lib/libginkgo_cuda.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libmuelu-adapters.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libmuelu-interface.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libmuelu.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libifpack2.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libanasazitpetra.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libModeLaplace.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libanasaziepetra.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libanasazi.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libmapvarlib.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libsuplib_cpp.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libsuplib_c.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libsuplib.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libsupes.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libaprepro_lib.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libio_info_lib.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libIonit.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libIotr.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libIohb.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libIogs.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libIogn.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libIovs.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libIoexo_fac.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libIofx.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libIoex.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libIoss.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libnemesis.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libexoIIv2for32.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libexodus_for.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libexodus.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libamesos2.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libbelosxpetra.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libbelostpetra.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libbelosepetra.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libbelos.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libml.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libifpack.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libzoltan2.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libamesos.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libgaleri-xpetra.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libgaleri-epetra.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libaztecoo.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libxpetra-sup.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libxpetra.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libepetraext.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libtrilinosss.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libtpetraext.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libtpetrainout.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libtpetra.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libkokkostsqr.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libtpetraclassiclinalg.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libtpetraclassicnodeapi.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libtpetraclassic.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libtriutils.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libzoltan.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libepetra.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libsacado.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libkokkoskernels.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libteuchoskokkoscomm.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libteuchoskokkoscompat.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libteuchosremainder.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libteuchosnumerics.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libteuchoscomm.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libteuchosparameterlist.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libteuchosparser.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libteuchoscore.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libkokkosalgorithms.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libkokkoscontainers.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libkokkoscore.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/trilinos-12.14.1-fxcxma77xpxavlvjw2xv3spehfuofqtr/lib/libgtest.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/matio-1.5.13-5mln2wjpgrtsziamla77qwq72lahu7lw/lib/libmatio.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/mumps-5.2.0-ogvpqpxl3u5fgcnopso7sgbzimdreh6n/lib/libdmumps.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/mumps-5.2.0-ogvpqpxl3u5fgcnopso7sgbzimdreh6n/lib/libmumps_common.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/mumps-5.2.0-ogvpqpxl3u5fgcnopso7sgbzimdreh6n/lib/libpord.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/suite-sparse-5.3.0-3khh7ltu7ywovhicuqqcildh7unimfgi/lib/libumfpack.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/suite-sparse-5.3.0-3khh7ltu7ywovhicuqqcildh7unimfgi/lib/libcholmod.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/suite-sparse-5.3.0-3khh7ltu7ywovhicuqqcildh7unimfgi/lib/libccolamd.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/suite-sparse-5.3.0-3khh7ltu7ywovhicuqqcildh7unimfgi/lib/libcolamd.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/suite-sparse-5.3.0-3khh7ltu7ywovhicuqqcildh7unimfgi/lib/libcamd.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/suite-sparse-5.3.0-3khh7ltu7ywovhicuqqcildh7unimfgi/lib/libsuitesparseconfig.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/suite-sparse-5.3.0-3khh7ltu7ywovhicuqqcildh7unimfgi/lib/libamd.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/adol-c-develop-vye5remz35fldbgvbs67yacgz7qolaoz/lib64/libadolc.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/arpack-ng-3.7.0-wlkb6xmmlpkz44y4n5mhkdxkr2zcbzc2/lib/libparpack.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/arpack-ng-3.7.0-wlkb6xmmlpkz44y4n5mhkdxkr2zcbzc2/lib/libarpack.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/assimp-4.0.1-pckw44k37ykr2ooyqw3svkv4nogsyxkw/lib/libassimp.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/gsl-2.5-dpk3nclhwe3kag4wbthw7vwnnij7tsus/lib/libgsl.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/gsl-2.5-dpk3nclhwe3kag4wbthw7vwnnij7tsus/lib/libgslcblas.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/muparser-2.2.6.1-zpzdrrgxsxk2bv5iqo7w2lncx2j3jyg4/lib/libmuparser.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/netcdf-cxx-4.2-kvbma4tctx5ptsxfiwju4vczfo36uo2k/lib/libnetcdf_c++.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/netcdf-4.7.0-rw2ctjf57qsvh2aqqg4zqltwmgjwa2xw/lib/libnetcdf.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKBO.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKBool.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKBRep.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKernel.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKFeat.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKFillet.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKG2d.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKG3d.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKGeomAlgo.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKGeomBase.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKHLR.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKIGES.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKMath.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKMesh.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKOffset.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKPrim.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKShHealing.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKSTEP.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKSTEPAttr.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKSTEPBase.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKSTEP209.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKSTL.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKTopAlgo.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/oce-0.18.3-zfy3xo3znsr3wyedp5szbjctk5hczs2d/lib/libTKXSBase.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/p4est-2.0-tpfehg54s3hssuvq7ki2htzuhppx44ss/lib/libp4est.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/p4est-2.0-tpfehg54s3hssuvq7ki2htzuhppx44ss/lib/libsc.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/netlib-scalapack-2.0.2-7nipopd4ozkp6n37dgqxzkwqssci7a5f/lib/libscalapack.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/slepc-3.11.0-7ztxwicpraemcw4cd4dpnmgvqqflg7d7/lib/libslepc.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/petsc-3.11.2-4kccpxhpqytkn7w2cmqq34cscb7yjdlb/lib/libpetsc.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/hypre-2.15.1-ylri7ab44qnsyyim4gqaylkgcz6omuv7/lib/libHYPRE.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/superlu-dist-6.1.1-rt74mda3l5yvcy3t3znvgbrkztcbdsz5/lib/libsuperlu_dist.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/openblas-0.3.6-kzlzpjmhmk7n5iqqaonvujph4rdrhghi/lib/libopenblas.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/hdf5-1.10.5-xa6rpqalfq4m25vjnz3nxukn2ccoaeta/lib/libhdf5hl_fortran.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/hdf5-1.10.5-xa6rpqalfq4m25vjnz3nxukn2ccoaeta/lib/libhdf5_fortran.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/hdf5-1.10.5-xa6rpqalfq4m25vjnz3nxukn2ccoaeta/lib/libhdf5_hl.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/hdf5-1.10.5-xa6rpqalfq4m25vjnz3nxukn2ccoaeta/lib/libhdf5.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/parmetis-4.0.3-mnpnut3zzzcjjdpifjtgfhlgygm6xtci/lib/libparmetis.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/metis-5.1.0-3wnvp4ji3wwu4v4vymszrhx6naehs6jc/lib/libmetis.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/zlib-1.2.11-5nus6knzumx4ik2yl44jxtgtsl7d54xb/lib/libz.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/mpich-3.3-j5u4l3i4w5xjawupwn4gsrb43tg6wntz/lib/libmpifort.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/mpich-3.3-j5u4l3i4w5xjawupwn4gsrb43tg6wntz/lib/libmpi.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/sundials-3.2.1-tbctfrvi7goy7hsxzip2bsjych4tpb3h/lib/libsundials_idas.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/sundials-3.2.1-tbctfrvi7goy7hsxzip2bsjych4tpb3h/lib/libsundials_arkode.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/sundials-3.2.1-tbctfrvi7goy7hsxzip2bsjych4tpb3h/lib/libsundials_kinsol.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/sundials-3.2.1-tbctfrvi7goy7hsxzip2bsjych4tpb3h/lib/libsundials_nvecserial.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/sundials-3.2.1-tbctfrvi7goy7hsxzip2bsjych4tpb3h/lib/libsundials_nvecparallel.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/symengine-0.4.0-iveahfqpvwkwlxvblg2uvkih4q7ufyfs/lib/libsymengine.so.0.4.0
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/gmp-6.1.2-qc4qcfz4monpllc3nqupdo7vwinf73sw/lib/libgmp.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/mpc-1.1.0-56lbd3hsdcjshbkwd6zgwztnmvmdvqsf/lib/libmpc.so
step-1: /usr/local/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/mpfr-4.0.1-dy5r7hirdgojscv4vf45t6fzusb66mu4/lib/libmpfr.so
step-1: CMakeFiles/step-1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dealii/git/testing-repo/dealii_examples/step-1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable step-1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/step-1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/step-1.dir/build: step-1

.PHONY : CMakeFiles/step-1.dir/build

CMakeFiles/step-1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/step-1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/step-1.dir/clean

CMakeFiles/step-1.dir/depend:
	cd /home/dealii/git/testing-repo/dealii_examples/step-1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dealii/git/testing-repo/dealii_examples/step-1 /home/dealii/git/testing-repo/dealii_examples/step-1 /home/dealii/git/testing-repo/dealii_examples/step-1 /home/dealii/git/testing-repo/dealii_examples/step-1 /home/dealii/git/testing-repo/dealii_examples/step-1/CMakeFiles/step-1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/step-1.dir/depend

