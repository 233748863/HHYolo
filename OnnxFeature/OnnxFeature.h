#pragma once

#include <windows.h>

#ifdef ONNXFEATURE_EXPORTS
#define ONNXAPI __declspec(dllexport)
#else
#define ONNXAPI __declspec(dllimport)
#endif

extern "C" {

/**
 * @brief 创建ONNX目标检测会话并加载模型
 * @param modelPath ONNX模型文件路径
 * @param classNamesPath 类名文件路径
 * @return 会话ID，成功返回>0，失败返回-1
 */
ONNXAPI int OnnxCreateSession(
    const char* modelPath,
    const char* classNamesPath
);

/**
 * @brief 创建ONNX OCR会话并加载模型
 * @param detectionModelPath OCR文本检测ONNX模型文件路径（必填）
 * @param recognitionModelPath OCR文字识别ONNX模型文件路径（必填）
 * @param classifierPath OCR方向分类器ONNX模型文件路径（可选，传入NULL或空字符串表示不使用）
 * @return OCR会话ID，成功返回>10000，失败返回-1
 */
ONNXAPI int OnnxCreateOcrSession(
    const char* detectionModelPath,
    const char* recognitionModelPath,
    const char* classifierPath
);

/**
 * @brief 使用会话ID对图像文件进行ONNX目标检测推理
 * @param sessionId 会话ID（由OnnxCreateSession返回）
 * @param imagePath 待检测图像文件路径
 * @param confidenceThreshold 置信度阈值 (0.0 - 1.0)
 * @return 检测结果字符串，格式为："类别名,置信度,x,y,宽度,高度|..."，
 *         未检测到目标返回"0"，发生错误返回"-1"
 */
ONNXAPI const char* OnnxDetectImage(
    int sessionId,
    const char* imagePath,
    double confidenceThreshold
);

/**
 * @brief 使用会话ID对指定窗口截图并进行ONNX目标检测推理
 * @param sessionId 会话ID（由OnnxCreateSession返回）
 * @param hwnd 目标窗口句柄
 * @param roiX 截图区域左上角X坐标（相对于窗口客户区）
 * @param roiY 截图区域左上角Y坐标（相对于窗口客户区）
 * @param roiWidth 截图区域宽度
 * @param roiHeight 截图区域高度
 * @param confidenceThreshold 置信度阈值 (0.0 - 1.0)
 * @return 检测结果字符串，格式为："类别名,置信度,x,y,宽度,高度|..."，
 *         未检测到目标返回"0"，发生错误返回"-1"
 */
ONNXAPI const char* OnnxDetectWindow(
    int sessionId,
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    double confidenceThreshold
);

/**
 * @brief 使用OCR会话ID对图像文件进行OCR文字识别
 * @param ocrSessionId OCR会话ID（由OnnxCreateOcrSession返回）
 * @param imagePath 待识别的图像文件路径
 * @param confidenceThreshold 置信度阈值 (0.0 - 1.0)
 * @return 识别结果字符串，格式为："文本1,x1,y1,w1,h1|文本2,x2,y2,w2,h2|..."，
 *         未识别到文本返回""，发生错误返回"-1"
 */
ONNXAPI const char* OnnxOcrImage(
    int ocrSessionId,
    const char* imagePath,
    double confidenceThreshold
);

/**
 * @brief 使用OCR会话ID对指定窗口截图并进行OCR文字识别
 * @param ocrSessionId OCR会话ID（由OnnxCreateOcrSession返回）
 * @param hwnd 目标窗口句柄
 * @param roiX 截图区域左上角X坐标（相对于窗口客户区）
 * @param roiY 截图区域左上角Y坐标（相对于窗口客户区）
 * @param roiWidth 截图区域宽度
 * @param roiHeight 截图区域高度
 * @param confidenceThreshold 置信度阈值 (0.0 - 1.0)
 * @return 识别结果字符串，格式为："文本1,x1,y1,w1,h1|文本2,x2,y2,w2,h2|..."，
 *         未识别到文本返回""，发生错误返回"-1"
 */
ONNXAPI const char* OnnxOcrWindow(
    int ocrSessionId,
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    double confidenceThreshold
);

/**
 * @brief PaddleOCR找字API - 在指定窗口区域查找特定文字
 * @param ocrSessionId OCR会话ID（由OnnxCreateOcrSession返回）
 * @param hwnd 目标窗口句柄
 * @param roiX 查找区域左上角X坐标（相对于窗口客户区）
 * @param roiY 查找区域左上角Y坐标（相对于窗口客户区）
 * @param roiWidth 查找区域宽度
 * @param roiHeight 查找区域高度
 * @param targetText 要查找的目标文字
 * @param confidenceThreshold 置信度阈值 (0.0 - 1.0)
 * @return 找到文字返回"x,y"坐标字符串，未找到返回"0"，发生错误返回"-1"
 */
ONNXAPI const char* PaddleOcrFindText(
    int ocrSessionId,
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    const char* targetText,
    double confidenceThreshold
);

/**
 * @brief PaddleOCR找字API - 在图像文件中查找特定文字
 * @param ocrSessionId OCR会话ID（由OnnxCreateOcrSession返回）
 * @param imagePath 图像文件路径
 * @param targetText 要查找的目标文字
 * @param confidenceThreshold 置信度阈值 (0.0 - 1.0)
 * @return 找到文字返回"x,y"坐标字符串，未找到返回"0"，发生错误返回"-1"
 */
ONNXAPI const char* PaddleOcrFindTextInImage(
    int ocrSessionId,
    const char* imagePath,
    const char* targetText,
    double confidenceThreshold
);


/**
 * @brief 清理ONNX资源
 */
ONNXAPI void OnnxPlugin_Cleanup();

/**
 * @brief 异步图像检测（非阻塞调用）
 * @param sessionId 会话ID
 * @param imagePath 图像文件路径
 * @param confidenceThreshold 置信度阈值
 * @return 任务ID，成功返回>0，失败返回-1
 */
ONNXAPI int OnnxDetectImageAsync(
    int sessionId,
    const char* imagePath,
    double confidenceThreshold
);

/**
 * @brief 异步窗口检测（非阻塞调用）
 * @param sessionId 会话ID
 * @param hwnd 目标窗口句柄
 * @param roiX 截图区域X坐标
 * @param roiY 截图区域Y坐标
 * @param roiWidth 截图区域宽度
 * @param roiHeight 截图区域高度
 * @param confidenceThreshold 置信度阈值
 * @return 任务ID，成功返回>0，失败返回-1
 */
ONNXAPI int OnnxDetectWindowAsync(
    int sessionId,
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    double confidenceThreshold
);

/**
 * @brief 异步OCR图像识别（非阻塞调用）
 * @param ocrSessionId OCR会话ID
 * @param imagePath 图像文件路径
 * @param confidenceThreshold 置信度阈值
 * @return 任务ID，成功返回>0，失败返回-1
 */
ONNXAPI int OnnxOcrImageAsync(
    int ocrSessionId,
    const char* imagePath,
    double confidenceThreshold
);

/**
 * @brief 异步OCR窗口识别（非阻塞调用）
 * @param ocrSessionId OCR会话ID
 * @param hwnd 目标窗口句柄
 * @param roiX 截图区域X坐标
 * @param roiY 截图区域Y坐标
 * @param roiWidth 截图区域宽度
 * @param roiHeight 截图区域高度
 * @param confidenceThreshold 置信度阈值
 * @return 任务ID，成功返回>0，失败返回-1
 */
ONNXAPI int OnnxOcrWindowAsync(
    int ocrSessionId,
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    double confidenceThreshold
);

/**
 * @brief 获取异步任务结果
 * @param taskId 任务ID
 * @param timeoutMs 超时时间（毫秒），0表示立即返回，-1表示无限等待
 * @return 检测结果字符串，任务未完成返回""，超时返回"timeout"，失败返回"-1"
 */
ONNXAPI const char* OnnxGetAsyncResult(
    int taskId,
    int timeoutMs
);

/**
 * @brief 取消异步任务
 * @param taskId 任务ID
 * @return 成功返回1，失败返回-1
 */
ONNXAPI int OnnxCancelAsyncTask(int taskId);

/**
 * @brief 获取性能统计信息
 * @param sessionId 会话ID，-1表示获取全局统计
 * @return 性能统计JSON字符串
 */
ONNXAPI const char* OnnxGetPerformanceStats(int sessionId);

/**
 * @brief 设置线程池大小
 * @param numThreads 线程数量，0表示自动检测CPU核心数
 * @return 成功返回1，失败返回-1
 */
ONNXAPI int OnnxSetThreadPoolSize(int numThreads);

} // extern "C"