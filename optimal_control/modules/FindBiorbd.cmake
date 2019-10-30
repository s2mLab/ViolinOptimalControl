# - Find Dlib
# Find the native Dlib includes and library
#
#  Biorbd_INCLUDE_DIR - where to find zlib.h, etc.
#  Biorbd_LIBRARY   - List of libraries when using zlib.
#  Biorbd_FOUND       - True if zlib found.

if (Biorbd_INCLUDE_DIR)
  # Already in cache, be silent
  set (Biorbd_FIND_QUIETLY TRUE)
endif (Biorbd_INCLUDE_DIR)

find_path (Biorbd_INCLUDE_DIR "BiorbdModel.h" PATHS ${CMAKE_INSTALL_PREFIX}/include/biorbd)
find_library (Biorbd_LIBRARY NAMES biorbd PATHS ${CMAKE_INSTALL_PREFIX}/lib/biorbd)

get_filename_component(Biorbd_LIB_PATH ${Biorbd_LIBRARY} DIRECTORY)
get_filename_component(Biorbd_LIB_NAME ${Biorbd_LIBRARY} NAME_WE)
get_filename_component(Biorbd_LIB_EXTENSION ${Biorbd_LIBRARY} EXT)
set(Biorbd_LIBRARIES
    ${Biorbd_LIB_PATH}/${Biorbd_LIB_NAME}${Biorbd_LIB_EXTENSION}
    ${Biorbd_LIB_PATH}/${Biorbd_LIB_NAME}_utils${Biorbd_LIB_EXTENSION}
    ${Biorbd_LIB_PATH}/${Biorbd_LIB_NAME}_rigidbody${Biorbd_LIB_EXTENSION}
    ${Biorbd_LIB_PATH}/${Biorbd_LIB_NAME}_muscles${Biorbd_LIB_EXTENSION}
    ${Biorbd_LIB_PATH}/${Biorbd_LIB_NAME}_actuators${Biorbd_LIB_EXTENSION}
)

# handle the QUIETLY and REQUIRED arguments and set DLIB_FOUND to TRUE if 
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (Biorbd DEFAULT_MSG 
  Biorbd_LIBRARIES
  Biorbd_INCLUDE_DIR)

