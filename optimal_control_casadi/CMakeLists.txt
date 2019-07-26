CMAKE_MINIMUM_REQUIRED ( VERSION 3.8 )
PROJECT("ViolonOptimalCasADi" CXX C)

# Set folder to look path to
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

# Set a default build type to 'Release' if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	MESSAGE(STATUS "Setting build type to 'Release' as none was specified.")
	SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
ENDIF()

find_package(CASADI REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(RBDL REQUIRED)
find_package(Dlib REQUIRED)
find_package(Biorbd REQUIRED)
link_directories(
  ${CMAKE_INSTALL_PREFIX}/lib
)

# Forward dynamics library
add_library(forward_dynamics_casadi SHARED src/forward_dynamics_casadi.cpp)
target_link_libraries(forward_dynamics_casadi
    ${Biorbd_LIBRARY}
    ${RBDL_LIBRARY}
)
target_include_directories(forward_dynamics_casadi PUBLIC
    include
    ${CMAKE_INSTALL_PREFIX}/include
    ${Biorbd_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${RBDL_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

# NLP related functions
add_library(utils SHARED src/utils.cpp)
target_include_directories(utils PUBLIC
    ${RBDL_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${Biorbd_INCLUDE_DIR}
    ${CASADI_INCLUDE_DIRS}
)

# Problem definitions
set(source_files
    eocar.cpp
    arm.cpp
)

# Setup each project
foreach(src_file ${source_files})
    get_filename_component(filename ${src_file} NAME_WE)

    # link_directories(${ACADO_LIBRARY_DIRS})
    ADD_EXECUTABLE (${filename}
        ${filename}.cpp
    )
    target_include_directories(${filename} PUBLIC
        src
        ${CASADI_INCLUDE_DIRS}
    )

    target_link_libraries(${filename}
        forward_dynamics_casadi
        utils
        ${CASADI_LIBRARIES}
      )
endforeach()

install(TARGETS forward_dynamics_casadi DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
