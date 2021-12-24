#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Darknet::dark" for configuration "RelWithDebInfo"
set_property(TARGET Darknet::dark APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(Darknet::dark PROPERTIES
  IMPORTED_IMPLIB_RELWITHDEBINFO "C:/Darknet_YOLOv4/darknet.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELWITHDEBINFO "CuDNN::CuDNN"
  IMPORTED_LOCATION_RELWITHDEBINFO "C:/Darknet_YOLOv4/darknet.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS Darknet::dark )
list(APPEND _IMPORT_CHECK_FILES_FOR_Darknet::dark "C:/Darknet_YOLOv4/darknet.lib" "C:/Darknet_YOLOv4/darknet.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
