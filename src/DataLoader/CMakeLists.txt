include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PUBLIC
                ${SOURCE_DIR}/DataLoader/CubeLUT.cpp
                ${SOURCE_DIR}/DataLoader/DataLoader.cpp
                )
