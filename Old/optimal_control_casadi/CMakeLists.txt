CMAKE_MINIMUM_REQUIRED ( VERSION 3.8 )
PROJECT("ViolonOptimalCasADi" CXX C)

# Set folder to look path to
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

## Set a default build type to 'Release' if none was specified
#IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
#	MESSAGE(STATUS "Setting build type to 'Release' as none was specified.")
#	SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
#  # Set the possible values of build type for cmake-gui
#  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
#ENDIF()

find_package(Casadi REQUIRED)
find_package(Boost REQUIRED)
find_package(RBDL REQUIRED)
find_package(Dlib REQUIRED)
find_package(Biorbd REQUIRED)
find_package(IPOPT REQUIRED)
find_package(Qt5 COMPONENTS Core Widgets Charts REQUIRED)
find_package(Eigen3 REQUIRED)

link_directories(
  ${CMAKE_INSTALL_PREFIX}/lib
)

# ConfigureProblem definitions
set(source_files
    eocarOpti.cpp
    eocarBiorbd.cpp
    UpAndDownBow.cpp
)

# library files
set(library_names
    utils
    AnimationCallback
)

# Setup each project
foreach(library_name ${library_names})
    # Forward dynamics library
    add_library(${library_name} SHARED src/${library_name}.cpp)
    target_link_libraries(${library_name}
        ${Biorbd_LIBRARIES}
        ${RBDL_LIBRARY}
        ${Qt5Core_LIBRARIES}
        ${Qt5Widgets_LIBRARIES}
        ${Qt5Charts_LIBRARIES}
    )
    target_include_directories(${library_name} PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_INSTALL_PREFIX}/include
        ${Biorbd_INCLUDE_DIR}
        ${RBDL_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${IPOPT_INCLUDE_DIR}
        ${Casadi_INCLUDE_DIR}
        ${Casadi_INCLUDE_DIR}/..
        ${Qt5Core_INCLUDE_DIRS}
        ${Qt5Widgets_INCLUDE_DIRS}
        ${Qt5Charts_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIR}
    )
endforeach()

# Setup each project
foreach(src_file ${source_files})
    get_filename_component(filename ${src_file} NAME_WE)

    # link_directories(${ACADO_LIBRARY_DIRS})
    ADD_EXECUTABLE (${filename}
        ${filename}.cpp
    )

    target_include_directories(${filename} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Casadi_INCLUDE_DIR}
        ${Biorbd_INCLUDE_DIR}
        ${Casadi_INCLUDE_DIR}
        ${Casadi_INCLUDE_DIR}/..
        ${RBDL_INCLUDE_DIR}
        ${Qt5Core_INCLUDE_DIRS}
        ${Qt5Widgets_INCLUDE_DIRS}
        ${Qt5Charts_INCLUDE_DIRS}
    )

    target_link_libraries(${filename}
        ${library_names}
        ${Biorbd_LIBRARIES}
        ${RBDL_LIBRARY}
        ${Casadi_LIBRARY}
        ${Qt5Core_LIBRARIES}
        ${Qt5Widgets_LIBRARIES}
        ${Qt5Charts_LIBRARIES}
      )
endforeach()

install(
    TARGETS ${library_names}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

