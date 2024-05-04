include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
if (MSVC)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(googletest)

add_executable(lut_loader_test
    LUTParserTest.cpp
    InputParamsTest.cpp
    InputArgsParserTest.cpp
    FileIOTest.cpp
    ImageProcessorTest.cpp
)

target_link_libraries(
  lut_loader_test
  ${MAIN_LIB_NAME}
  GTest::gtest_main
  gmock_main
  ${LIBRARIES_TO_LINK}
)

target_include_directories(lut_loader_test PUBLIC ${SOURCE_DIR} ${INCLUDE_DIR} ${TEST_DIR})

include(GoogleTest)
gtest_discover_tests(lut_loader_test)
