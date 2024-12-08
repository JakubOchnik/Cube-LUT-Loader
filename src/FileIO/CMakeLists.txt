include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
                ${SOURCE_DIR}/FileIO/CubeLUT.cpp
                ${SOURCE_DIR}/FileIO/FileIO.cpp
                ${SOURCE_DIR}/FileIO/ImageWrapper.cpp
                )
