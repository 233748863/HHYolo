# TC - 图像处理与OCR工具库

一个基于OpenCV和Tesseract的Windows图像处理DLL库，提供图像匹配、OCR识别、屏幕截图和内存读取等功能。

## 功能特性

### 图像处理
- **图像模板匹配**：支持文件和内存中的图像匹配
- **窗口截图**：对指定窗口区域进行截图
- **多模板搜索**：一次性搜索多个模板图像
- **颜色匹配**：支持模糊颜色匹配和偏色设置

### OCR识别
- **文字识别**：基于Tesseract的OCR功能
- **文本定位**：查找特定文本并返回坐标
- **多语言支持**：支持中文、英文等多种语言

### 游戏辅助功能
- **网格分析**：基于锚点的批量UI元素分析
- **内存读取**：读取目标进程内存数据
- **颜色检测**：精确的颜色匹配和偏色处理

## 项目结构

```
TC/
├── HHYolo/          # 图像处理和OCR功能模块
│   ├── HHYolo.h     # 头文件 - 函数声明
│   ├── HHYolo.cpp   # 实现文件
│   └── HHYolo.def   # 导出定义
├── OnnxFeature/     # ONNX模型相关功能
│   ├── OnnxFeature.h
│   ├── OnnxFeature.cpp
│   └── OnnxFeature.def
├── Common/          # 公共工具模块
│   ├── Utils.h
│   └── Utils.cpp
├── CMakeLists.txt   # CMake构建配置
└── README.md        # 项目说明
```

## 编译说明

### 依赖库
- OpenCV 4.x
- Tesseract OCR
- Windows SDK

### 构建步骤
1. 安装CMake和Visual Studio
2. 配置OpenCV和Tesseract环境变量
3. 运行CMake生成项目文件
4. 使用Visual Studio编译

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## API概览

### 图像匹配
```cpp
// 文件路径匹配
int FindImage(const char* imgPath, const char* tplPath, double similarity, int* x, int* y, double* confidence);

// 内存匹配
int FindImageFromMem(const unsigned char* imgData, int imgWidth, int imgHeight, int imgChannels, 
                    const char* tplPath, double similarity, int* x, int* y, double* confidence);

// 窗口截图匹配
int CaptureAndFindImage(HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight,
                       const char* tplPath, double similarity, int* matchX, int* matchY, double* confidence);
```

### OCR功能
```cpp
// 区域OCR识别
const char* Ocr(HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight,
               const char* lang, const char* tessdataPath);

// 文件OCR识别
const char* OcrFile(const char* imgPath, const char* lang, const char* tessdataPath);

// 文本定位
const char* HH_FindText(HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight,
                       const char* textToFind, const char* lang, const char* tessdataPath);
```

### 高级功能
```cpp
// 内存读取
const char* ReadMemory(HWND hwnd, const char* addressExpression, const char* readType, int readSize);

// 网格分析
const char* AnalyzeGrid(HWND hwnd, const char* anchorTplPath, double similarity,
                       int gridRows, int gridCols, ...);

// 颜色匹配
const char* FindColor(HWND hwnd, int x, int y, const char* colorStr);
```

## 使用示例

### 基本图像匹配
```cpp
#include "HHYolo.h"

int main() {
    int x, y;
    double confidence;
    
    int result = FindImage("main.png", "template.png", 0.8, &x, &y, &confidence);
    
    if (result == 1) {
        printf("找到匹配，坐标: (%d, %d), 置信度: %.2f\n", x, y, confidence);
    }
    
    return 0;
}
```

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系

如有问题请通过GitHub Issues联系我们。