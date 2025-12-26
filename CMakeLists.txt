cmake_minimum_required(VERSION 3.16)
set(VCPKG_TARGET_TRIPLET "x86-windows-static")
project(TC_Solution LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ========== 强制只使用 Release 配置 ==========
set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "构建类型" FORCE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "仅支持 Release 配置，请使用 --config Release 参数")
endif()

# ========== 简化运行时库设置（仅 Release）==========
if(MSVC)
    # 只使用静态运行时（/MT）
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
    add_compile_options(/MT)
    add_link_options(/NODEFAULTLIB:msvcrt.lib /IGNORE:4098)
    message(STATUS "已配置 Release 静态链接版本")
endif()

# ========== 主项目：HHYolo.dll（完全静态链接）==========
# HHYolo.dll 使用静态链接方式，包含所有依赖库，无需外部DLL依赖
set(VCPKG_TARGET_TRIPLET "x86-windows-static")
set(VCPKG_HOST_TRIPLET "x86-windows-static")

# 查找静态版本的OpenCV和Tesseract库
find_package(quirc CONFIG REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(Tesseract REQUIRED)

# ========== Common 静态库 ==========
add_library(Common STATIC
    Common/Utils.cpp
    Common/Utils.h
)
target_include_directories(Common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Common)
target_link_libraries(Common PUBLIC
    opencv_core opencv_imgproc opencv_imgcodecs
    user32 gdi32
)

# 创建HHYolo动态链接库
add_library(HHYolo SHARED
    HHYolo/HHYolo.cpp
    HHYolo/HHYolo.def
    HHYolo/HHYolo.h
)

# 包含头文件目录
target_include_directories(HHYolo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/HHYolo)

# 链接静态库 - HHYolo.dll将包含所有依赖，无需外部DLL
target_link_libraries(HHYolo PRIVATE
    Common
    Tesseract::libtesseract
    opencv_core opencv_imgproc opencv_imgcodecs
    user32
)

# 编译定义 - 确保使用静态库版本
target_compile_definitions(HHYolo PRIVATE
    HHYOLO_EXPORTS
    OPENCV_STATIC_LIB
    _STATIC_LIBTESSERACT
)

# ========== 插件：OnnxFeature.dll（混合链接）==========
# OnnxFeature.dll 使用混合链接方式：
# - OpenCV: 静态链接（包含在DLL内部）
# - ONNX Runtime: 动态链接（需要外部onnxruntime.dll）

# 手动查找动态版本的ONNX Runtime库
# 优先使用环境变量，其次使用vcpkg默认路径
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_SEARCH_PATHS "$ENV{ONNXRUNTIME_ROOT}/include" "$ENV{ONNXRUNTIME_ROOT}/lib")
else()
    set(ONNXRUNTIME_SEARCH_PATHS 
        "${VCPKG_INSTALLED_DIR}/x86-windows/include"
        "C:/vcpkg/installed/x86-windows/include"
    )
    set(ONNXRUNTIME_LIB_PATHS
        "${VCPKG_INSTALLED_DIR}/x86-windows/lib"
        "C:/vcpkg/installed/x86-windows/lib"
    )
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS ${ONNXRUNTIME_SEARCH_PATHS}
    REQUIRED
)

find_library(ONNXRUNTIME_LIB onnxruntime.lib
    PATHS ${ONNXRUNTIME_LIB_PATHS}
    REQUIRED
)

# 创建OnnxFeature动态链接库
add_library(OnnxFeature SHARED
    OnnxFeature/OnnxFeature.cpp
    OnnxFeature/OnnxFeature.def
)

# 包含头文件目录 - 包含HHYolo头文件以访问OCR功能
target_include_directories(OnnxFeature PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/HHYolo
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# 混合链接库：
# - ONNX Runtime: 动态链接（导入库）
# - OpenCV: 静态链接（包含在DLL内部）
target_link_libraries(OnnxFeature PRIVATE
    Common
    ${ONNXRUNTIME_LIB}
    opencv_core opencv_imgproc opencv_imgcodecs opencv_dnn
)

# 编译定义 - 确保OpenCV使用静态版本
target_compile_definitions(OnnxFeature PRIVATE 
    ONNXPLUGIN_EXPORTS
    OPENCV_STATIC_LIB
)

# 依赖关系 - OnnxFeature依赖于HHYolo的构建
add_dependencies(OnnxFeature HHYolo)

# 查找 onnxruntime.dll 路径
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_DLL_PATH "$ENV{ONNXRUNTIME_ROOT}/bin/onnxruntime.dll")
elseif(EXISTS "${VCPKG_INSTALLED_DIR}/x86-windows/bin/onnxruntime.dll")
    set(ONNXRUNTIME_DLL_PATH "${VCPKG_INSTALLED_DIR}/x86-windows/bin/onnxruntime.dll")
else()
    set(ONNXRUNTIME_DLL_PATH "C:/vcpkg/installed/x86-windows/bin/onnxruntime.dll")
endif()

# 复制 onnxruntime.dll 到输出目录
# OnnxFeature.dll需要外部onnxruntime.dll才能正常运行
add_custom_command(TARGET OnnxFeature POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL_PATH}"
        $<TARGET_FILE_DIR:OnnxFeature>
)