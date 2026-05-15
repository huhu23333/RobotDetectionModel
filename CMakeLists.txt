cmake_minimum_required(VERSION 3.16)
project(ArmorDetectionModel)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

# ============================================================
# 查找依赖包
# ============================================================

# OpenVINO
find_package(OpenVINO REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 海康 MVS SDK
find_library(MVCAMERA_LIB MvCameraControl PATHS /opt/MVS/lib/64)
include_directories(/opt/MVS/include)

# 线程
find_package(Threads REQUIRED)

# ============================================================
# 编译选项
# ============================================================
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")

# ============================================================
# 源文件
# ============================================================
set(SOURCES
    src/main.cpp
    OpenvinoInfer.cpp
)

# ============================================================
# 可执行文件
# ============================================================
add_executable(armor_detection ${SOURCES})

target_link_libraries(armor_detection
    ${OpenCV_LIBS}
    openvino::runtime
    ${MVCAMERA_LIB}
    pthread
)

# 链接 OpenVINO 库目录
target_link_directories(armor_detection PRIVATE
    /opt/MVS/lib/64
)

# ============================================================
# 安装规则
# ============================================================
install(TARGETS armor_detection RUNTIME DESTINATION bin)
install(DIRECTORY Model/ DESTINATION bin/Model)

# 项目自身头文件目录（包含 OpenvinoInfer.h）
include_directories(${PROJECT_SOURCE_DIR})
