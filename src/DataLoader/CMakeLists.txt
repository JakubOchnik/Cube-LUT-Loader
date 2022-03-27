include_directories(${INCLUDE_DIR})

target_sources(${PROJECT_NAME} PUBLIC 
                ${SOURCE_DIR}/DataLoader/CubeLUT.cpp
                ${SOURCE_DIR}/DataLoader/DataLoader.cpp
                )