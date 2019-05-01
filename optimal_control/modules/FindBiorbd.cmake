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

find_path (Biorbd_INCLUDE_DIR "s2mMusculoSkeletalModel.h" PATHS ${CMAKE_INSTALL_PREFIX}/include/biorbd/include)
find_library (Biorbd_LIBRARY NAMES biorbd PATHS ${CMAKE_INSTALL_PREFIX}/lib/biorbd)

# handle the QUIETLY and REQUIRED arguments and set DLIB_FOUND to TRUE if 
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (Biorbd DEFAULT_MSG 
  Biorbd_LIBRARY 
  Biorbd_INCLUDE_DIR)

mark_as_advanced (Biorbd_LIBRARY Biorbd_INCLUDE_DIR)
