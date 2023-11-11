include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
                ${SOURCE_DIR}/TaskDispatcher/TaskDispatcher.cpp
                ${SOURCE_DIR}/TaskDispatcher/InputParams.cpp
                )
