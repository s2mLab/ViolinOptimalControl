### DO NOT MODIFY THIS PART ###

CMAKE_MINIMUM_REQUIRED ( VERSION 3.1 )
PROJECT("acado" CXX C)

# Set folder to look path to
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

# Set a default build type to 'Release' if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	MESSAGE(STATUS "Setting build type to 'Release' as none was specified.")
	SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
ENDIF()

find_package(ACADO REQUIRED)
find_package(Boost REQUIRED)
set(EIGEN3_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/acado/external_packages/eigen3)
find_package(RBDL REQUIRED)
find_package(Dlib REQUIRED)
find_package(Biorbd REQUIRED)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

# Write down all files to include here
set(source_files 
    main_acado.cpp
) 

# Setup each project
foreach(src_file ${source_files})
    get_filename_component(filename ${src_file} NAME_WE)

    # link_directories(${ACADO_LIBRARY_DIRS})
    ADD_EXECUTABLE (  ${filename}
        ${filename}.cpp
    )
    target_include_directories(${filename} PUBLIC 
      ${ACADO_INCLUDE_DIRS} ${ACADO_INCLUDE_DIRS}/..
      ${Biorbd_INCLUDE_DIR}
      ${EIGEN3_INCLUDE_DIR}
      ${RBDL_INCLUDE_DIR}
      ${Boost_INCLUDE_DIRS}
    )
    
    target_link_libraries(${filename} 
      ${Biorbd_LIBRARY}
      ${RBDL_LIBRARY}
      ${ACADO_SHARED_LIBRARIES}
      )
endforeach()


