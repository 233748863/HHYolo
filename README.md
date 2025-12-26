# TC - 图像处理与OCR工具库

一个基于OpenCV和Tesseract的Windows图像处理DLL库，提供图像匹配、OCR识别、屏幕截图和内存读取等功能。

## 功能特性

### 图像处理 (HHYolo)
- **图像模板匹配**：支持文件和内存中的图像匹配
- **窗口截图**：对指定窗口区域进行截图
- **多模板搜索**：一次性搜索多个模板图像
- **颜色匹配**：支持模糊颜色匹配和偏色设置

### OCR识别 (HHYolo - Tesseract)
- **文字识别**：基于Tesseract的OCR功能
- **文本定位**：查找特定文本并返回坐标
- **多语言支持**：支持中文、英文等多种语言

### ONNX推理 (OnnxFeature)
- **目标检测**：基于YOLOv11的ONNX模型推理
- **PaddleOCR**：基于PaddleOCR的文字识别
- **异步处理**：支持非阻塞的异步检测和识别
- **会话管理**：支持多模型会话复用

### 游戏辅助功能
- **网格分析**：基于锚点的批量UI元素分析
- **内存读取**：读取目标进程内存数据（支持32位和64位进程）
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
- ONNX Runtime
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

### HHYolo 模块

#### 图像匹配
```cpp
// 文件路径匹配
int FindImage(const char* imgPath, const char* tplPath, double similarity, int* x, int* y, double* confidence);

// 内存匹配
int FindImageFromMem(const unsigned char* imgData, int imgWidth, int imgHeight, int imgChannels, 
                    const char* tplPath, double similarity, int* x, int* y, double* confidence);

// 窗口截图匹配（单个）
int CaptureAndFindImage(HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight,
                       const char* tplPath, double similarity, int* matchX, int* matchY, double* confidence);

// 窗口截图匹配（所有匹配项）
// 返回: "数量;x1,y1,sim1|x2,y2,sim2|..."
const char* CaptureAndFindAllImages(HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight,
                                   const char* tplPath, double similarity);

// 多模板同时搜索
// multiTplPaths: 多个模板路径用"|"分隔
// 返回: "文件名1,x,y,置信度|文件名2,x,y,置信度|..."
const char* CaptureAndFindMultiTemplates(HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight,
                                        const char* multiTplPaths, double similarity);
```

#### OCR功能 (Tesseract)
```cpp
// 区域OCR识别
// 返回: "文本1,x1,y1|文本2,x2,y2|..."
const char* Ocr(HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight,
               const char* lang, const char* tessdataPath);

// 文件OCR识别
const char* OcrFile(const char* imgPath, const char* lang, const char* tessdataPath);

// 文本定位
// 返回: "x,y" 或 "0"(未找到) 或 "-1"(错误)
const char* HH_FindText(HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight,
                       const char* textToFind, const char* lang, const char* tessdataPath);
```

#### 高级功能
```cpp
// 内存读取（支持32位和64位目标进程）
// addressExpression 格式: "模块名+偏移[+偏移2[+偏移3...]]"
// readType: "i"(整型), "f"(浮点), "d"(双精度), "s"(字符串), "w"(宽字符串)
const char* ReadMemory(HWND hwnd, const char* addressExpression, const char* readType, int readSize);

// 网格分析（基于锚点的批量UI分析）
// 返回: "centerX1,centerY1,health1|centerX2,centerY2,health2|..."
const char* AnalyzeGrid(HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight,
                       const char* anchorTplPath, double similarity,
                       int gridRows, int gridCols,
                       int firstCellOffsetX, int firstCellOffsetY,
                       int cellWidth, int cellHeight,
                       int horizontalGap, int verticalGap,
                       int healthBarOffsetX, int healthBarOffsetY,
                       int healthBarWidth, int healthBarHeight);

// 颜色匹配
// colorStr 格式: "RRGGBB-偏差" 或 "RRGGBB-偏差R,偏差G,偏差B"，多个用"|"分隔
// 返回: 匹配的颜色索引 "1|3|..." 或 "0"(未找到)
const char* FindColor(HWND hwnd, int x, int y, const char* colorStr);
```

### OnnxFeature 模块

#### 会话管理
```cpp
// 创建目标检测会话
// 返回: 会话ID (>0成功, -1失败)
int OnnxCreateSession(const char* modelPath, const char* classNamesPath);

// 创建OCR会话 (PaddleOCR)
// classifierPath 可选，传NULL表示不使用方向分类器
// 返回: OCR会话ID (>10000成功, -1失败)
int OnnxCreateOcrSession(const char* detectionModelPath, const char* recognitionModelPath, const char* classifierPath);

// 清理所有ONNX资源
void OnnxPlugin_Cleanup();
```

#### 目标检测
```cpp
// 图像文件检测
// 返回: "类别名,置信度,x,y,宽度,高度|..." 或 "0"(未检测到) 或 "-1"(错误)
const char* OnnxDetectImage(int sessionId, const char* imagePath, double confidenceThreshold);

// 窗口截图检测
const char* OnnxDetectWindow(int sessionId, HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight, double confidenceThreshold);
```

#### PaddleOCR
```cpp
// 图像文件OCR
// 返回: "文本1,x1,y1,w1,h1|文本2,x2,y2,w2,h2|..."
const char* OnnxOcrImage(int ocrSessionId, const char* imagePath, double confidenceThreshold);

// 窗口截图OCR
const char* OnnxOcrWindow(int ocrSessionId, HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight, double confidenceThreshold);

// 找字（窗口）
// 返回: "x,y" 或 "0"(未找到) 或 "-1"(错误)
const char* PaddleOcrFindText(int ocrSessionId, HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight, const char* targetText, double confidenceThreshold);

// 找字（图像文件）
const char* PaddleOcrFindTextInImage(int ocrSessionId, const char* imagePath, const char* targetText, double confidenceThreshold);
```

#### 异步API
```cpp
// 异步检测（返回任务ID）
int OnnxDetectImageAsync(int sessionId, const char* imagePath, double confidenceThreshold);
int OnnxDetectWindowAsync(int sessionId, HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight, double confidenceThreshold);
int OnnxOcrImageAsync(int ocrSessionId, const char* imagePath, double confidenceThreshold);
int OnnxOcrWindowAsync(int ocrSessionId, HWND hwnd, int roiX, int roiY, int roiWidth, int roiHeight, double confidenceThreshold);

// 获取异步结果
// timeoutMs: 0=立即返回, -1=无限等待
// 返回: 结果字符串 或 ""(未完成) 或 "timeout"(超时) 或 "-1"(错误)
const char* OnnxGetAsyncResult(int taskId, int timeoutMs);

// 取消异步任务
int OnnxCancelAsyncTask(int taskId);
```

#### 性能管理
```cpp
// 获取性能统计（JSON格式）
const char* OnnxGetPerformanceStats(int sessionId);

// 设置线程池大小（0=自动检测CPU核心数）
int OnnxSetThreadPoolSize(int numThreads);
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

### 内存读取示例
```cpp
#include "HHYolo.h"

int main() {
    HWND hwnd = FindWindow(NULL, "目标窗口");
    
    // 读取32位或64位进程的内存
    // 格式: 模块名+基址偏移[+指针偏移1[+指针偏移2...]]
    const char* result = ReadMemory(hwnd, "Game.exe+AABBCC+10+20", "i", 4);
    
    if (strcmp(result, "-1") != 0) {
        printf("读取到的值: %s\n", result);
    }
    
    return 0;
}
```

### ONNX目标检测示例
```cpp
#include "OnnxFeature.h"

int main() {
    // 创建检测会话
    int sessionId = OnnxCreateSession("yolov11.onnx", "classes.txt");
    if (sessionId < 0) {
        printf("创建会话失败\n");
        return -1;
    }
    
    // 检测图像
    const char* result = OnnxDetectImage(sessionId, "test.png", 0.5);
    printf("检测结果: %s\n", result);
    
    // 清理资源
    OnnxPlugin_Cleanup();
    return 0;
}
```

### PaddleOCR示例
```cpp
#include "OnnxFeature.h"

int main() {
    // 创建OCR会话
    int ocrId = OnnxCreateOcrSession("det.onnx", "rec.onnx", NULL);
    if (ocrId < 0) {
        printf("创建OCR会话失败\n");
        return -1;
    }
    
    // 识别图像中的文字
    const char* result = OnnxOcrImage(ocrId, "screenshot.png", 0.5);
    printf("识别结果: %s\n", result);
    
    // 查找特定文字
    const char* pos = PaddleOcrFindTextInImage(ocrId, "screenshot.png", "确定", 0.5);
    if (strcmp(pos, "0") != 0 && strcmp(pos, "-1") != 0) {
        printf("找到文字位置: %s\n", pos);
    }
    
    OnnxPlugin_Cleanup();
    return 0;
}
```

## 技术说明

### 内存读取
- 32位DLL可同时读取32位和64位目标进程
- 64位进程通过 `NtWow64ReadVirtualMemory64` API 实现跨位数读取
- 支持多级指针自动解析

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系

如有问题请通过GitHub Issues联系我们。