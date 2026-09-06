add_library(xenia-nui STATIC)
xe_platform_sources(xenia-nui ${CMAKE_CURRENT_SOURCE_DIR} RECURSIVE)
target_link_libraries(xenia-nui PUBLIC xenia-base xenia-hid fmt)
# .nuirec depth/colour compression (nui_recorder.cc).
target_link_libraries(xenia-nui PRIVATE zstd)
xe_target_defaults(xenia-nui)

if(XENIA_BUILD_TESTS)
  set(CMAKE_FOLDER "tests")
  add_subdirectory(testing)
endif()

if(WIN32)
  # Media Foundation webcam capture (camera_capture_win.cc).
  target_link_libraries(xenia-nui PRIVATE mfplat mfreadwrite mfuuid mf ole32)
endif()

if(WIN32)
  # Console tool for the webcam source: lists cameras, captures frames, runs
  # pose estimation or the whole WebcamNuiSource and writes PNG dumps.
  # nui_webcam_demo.cc is excluded from the library by the *_demo.cc rule.
  add_executable(xenia-nui-webcam-demo
    ${CMAKE_CURRENT_SOURCE_DIR}/nui_webcam_demo.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_win.cc
  )
  target_link_libraries(xenia-nui-webcam-demo PRIVATE fmt xenia-base xenia-nui)
  xe_target_defaults(xenia-nui-webcam-demo)
endif()
