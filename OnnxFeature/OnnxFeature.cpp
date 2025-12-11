#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <windows.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <opencv2/opencv.hpp>
#include "../Common/Utils.h"

// 检测结果结构体
struct Detection {
    std::string className;
    double confidence;
    int x, y, width, height;
};

// 全局变量用于存储返回字符串
static thread_local char g_resultBuffer[8192];

// 全局ONNX环境变量，用于管理ONNX Runtime资源
static std::unique_ptr<Ort::Env> g_onnxEnv = nullptr;

// 会话管理相关全局变量
static std::map<int, std::unique_ptr<class OnnxSession>> g_sessions;
static std::map<int, std::unique_ptr<class OcrSession>> g_ocrSessions;
static int g_nextSessionId = 1;
static int g_nextOcrSessionId = 10000; // OCR会话ID从10000开始，避免与普通会话冲突
static std::mutex g_sessionMutex;

// ========== 异步任务管理 ==========

// 异步任务状态枚举
enum class AsyncTaskStatus {
    PENDING,
    RUNNING,
    COMPLETED,
    CANCELLED,
    FAILED
};

// 异步任务结构体
struct AsyncTask {
    int taskId;
    AsyncTaskStatus status;
    std::string result;
    std::future<std::string> future;
    std::chrono::steady_clock::time_point startTime;
    std::string taskType; // "detect_image", "detect_window", "ocr_image", "ocr_window"
    int sessionId;
};

// 异步任务管理相关全局变量
static std::map<int, std::unique_ptr<AsyncTask>> g_asyncTasks;
static int g_nextTaskId = 1;
static std::mutex g_taskMutex;
static std::unique_ptr<cv::Mat> g_lastPreprocessedImage = nullptr;
static std::mutex g_preprocessMutex;

// 线程池管理
static std::unique_ptr<class ThreadPool> g_threadPool = nullptr;
static int g_threadPoolSize = 0; // 0表示自动检测
static std::mutex g_threadPoolMutex;

// 性能统计结构
struct PerformanceStats {
    int totalTasks = 0;
    int completedTasks = 0;
    int failedTasks = 0;
    int cancelledTasks = 0;
    double avgProcessingTimeMs = 0.0;
    std::chrono::steady_clock::time_point lastResetTime;
    std::map<std::string, int> taskTypeCounts;
};

static PerformanceStats g_performanceStats;
static std::mutex g_statsMutex;

// 线程池类
class ThreadPool {
public:
    ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stop) return;
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }
    
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

// 初始化线程池
static void InitializeThreadPool() {
    std::lock_guard<std::mutex> lock(g_threadPoolMutex);
    if (!g_threadPool) {
        int numThreads = g_threadPoolSize;
        if (numThreads <= 0) {
            // 自动检测CPU核心数
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            numThreads = sysInfo.dwNumberOfProcessors;
            if (numThreads <= 0) numThreads = 4; // 默认4个线程
        }
        g_threadPool = std::make_unique<ThreadPool>(numThreads);
    }
}

// ========== 前向声明 ==========
static cv::Mat LetterboxResize(const cv::Mat& image, int targetWidth, int targetHeight, 
                                cv::Scalar color = cv::Scalar(114, 114, 114));

// 图像预处理缓存类
class ImagePreprocessor {
public:
    struct PreprocessedImage {
        cv::Mat image;
        std::string imagePath;
        std::chrono::steady_clock::time_point timestamp;
        int width = 0;
        int height = 0;
    };
    
    static PreprocessedImage PreprocessAndCache(const std::string& imagePath, int targetWidth, int targetHeight) {
        std::lock_guard<std::mutex> lock(g_preprocessMutex);
        
        // 检查缓存
        auto it = g_imageCache.find(imagePath);
        if (it != g_imageCache.end()) {
            auto& cached = it->second;
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - cached.timestamp);
            
            // 缓存有效期为5分钟
            if (duration.count() < 300 && cached.width == targetWidth && cached.height == targetHeight) {
                return cached;
            }
        }
        
        // 预处理图像
        PreprocessedImage result;
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            return result;
        }
        
        result.image = LetterboxResize(image, targetWidth, targetHeight);
        result.imagePath = imagePath;
        result.timestamp = std::chrono::steady_clock::now();
        result.width = targetWidth;
        result.height = targetHeight;
        
        // 更新缓存
        g_imageCache[imagePath] = result;
        
        // 限制缓存大小
        if (g_imageCache.size() > 10) { // 最多缓存10张图像
            auto oldest = g_imageCache.begin();
            for (auto it = g_imageCache.begin(); it != g_imageCache.end(); ++it) {
                if (it->second.timestamp < oldest->second.timestamp) {
                    oldest = it;
                }
            }
            g_imageCache.erase(oldest);
        }
        
        return result;
    }
    
    static void ClearCache() {
        std::lock_guard<std::mutex> lock(g_preprocessMutex);
        g_imageCache.clear();
    }
    
private:
    static std::map<std::string, PreprocessedImage> g_imageCache;
};

std::map<std::string, ImagePreprocessor::PreprocessedImage> ImagePreprocessor::g_imageCache;


// 读取类名文件
static std::vector<std::string> ReadClassNames(const char* classNamesPath) {
    std::vector<std::string> classNames;
    if (!classNamesPath || !*classNamesPath) return classNames;
    
    std::ifstream file(classNamesPath);
    if (!file.is_open()) return classNames;
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            classNames.push_back(line);
        }
    }
    
    return classNames;
}

// ========== 辅助函数 ==========

// 加载 PaddleOCR 字典文件
static std::vector<std::string> LoadOcrDictionary(const char* dictPath) {
    std::vector<std::string> dictionary;
    dictionary.push_back(" "); // 空白字符,CTC blank token
    
    if (!dictPath || !*dictPath || !Common::FileExists(dictPath)) {
        // 如果没有字典文件,使用默认的简化字符集
        return dictionary;
    }
    
    std::ifstream file(dictPath);
    if (!file.is_open()) return dictionary;
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            dictionary.push_back(line);
        }
    }
    
    return dictionary;
}

// CTC 贪婪解码
static std::string CtcGreedyDecoder(const float* outputData, const std::vector<int64_t>& outputShape, 
                                     const std::vector<std::string>& dictionary) {
    if (outputShape.size() < 2) return "";
    
    int timeSteps = static_cast<int>(outputShape[1]);
    int numClasses = static_cast<int>(outputShape[2]);
    
    std::string result;
    int lastIndex = -1;
    
    for (int t = 0; t < timeSteps; ++t) {
        // 找到当前时间步的最大概率索引
        int maxIndex = 0;
        float maxProb = outputData[t * numClasses];
        
        for (int c = 1; c < numClasses; ++c) {
            float prob = outputData[t * numClasses + c];
            if (prob > maxProb) {
                maxProb = prob;
                maxIndex = c;
            }
        }
        
        // CTC 解码规则: 跳过 blank (索引0) 和重复字符
        if (maxIndex != 0 && maxIndex != lastIndex) {
            if (maxIndex < static_cast<int>(dictionary.size())) {
                result += dictionary[maxIndex];
            }
        }
        
        lastIndex = maxIndex;
    }
    
    return result;
}

// Letterbox 缩放 (保持长宽比)
static cv::Mat LetterboxResize(const cv::Mat& image, int targetWidth, int targetHeight, 
                                cv::Scalar color) {
    int imgWidth = image.cols;
    int imgHeight = image.rows;
    
    // 计算缩放比例
    float scale = std::min(static_cast<float>(targetWidth) / imgWidth, 
                           static_cast<float>(targetHeight) / imgHeight);
    
    int newWidth = static_cast<int>(imgWidth * scale);
    int newHeight = static_cast<int>(imgHeight * scale);
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newWidth, newHeight));
    
    // 创建目标尺寸的画布并填充
    cv::Mat result(targetHeight, targetWidth, image.type(), color);
    
    // 计算居中位置
    int top = (targetHeight - newHeight) / 2;
    int left = (targetWidth - newWidth) / 2;
    
    // 将缩放后的图像复制到画布中心
    resized.copyTo(result(cv::Rect(left, top, newWidth, newHeight)));
    
    return result;
}

// NMS (非极大值抑制)
static std::vector<Detection> ApplyNMS(const std::vector<Detection>& detections, float iouThreshold) {
    if (detections.empty()) return {};
    
    // 按置信度降序排序
    std::vector<Detection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(), 
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    
    std::vector<Detection> result;
    std::vector<bool> suppressed(sorted.size(), false);
    
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(sorted[i]);
        
        // 计算与后续框的 IOU
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (suppressed[j]) continue;
            
            // 计算交集
            int x1 = std::max(sorted[i].x, sorted[j].x);
            int y1 = std::max(sorted[i].y, sorted[j].y);
            int x2 = std::min(sorted[i].x + sorted[i].width, sorted[j].x + sorted[j].width);
            int y2 = std::min(sorted[i].y + sorted[i].height, sorted[j].y + sorted[j].height);
            
            int intersectionWidth = std::max(0, x2 - x1);
            int intersectionHeight = std::max(0, y2 - y1);
            int intersectionArea = intersectionWidth * intersectionHeight;
            
            // 计算并集
            int area1 = sorted[i].width * sorted[i].height;
            int area2 = sorted[j].width * sorted[j].height;
            int unionArea = area1 + area2 - intersectionArea;
            
            // 计算 IOU
            float iou = static_cast<float>(intersectionArea) / unionArea;
            
            if (iou > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

// OCR文本框结构体
struct TextBox {
    std::string text;
    int x, y, width, height;
    double confidence;
};

// ONNX会话类
class OnnxSession {
public:
    OnnxSession(const char* modelPath, const char* classNamesPath) {
        // 检查文件是否存在
        if (!Common::FileExists(modelPath)) {
            m_valid = false;
            return;
        }
        
        // 读取类名文件
        m_classNames = ReadClassNames(classNamesPath);
        if (m_classNames.empty()) {
            m_valid = false;
            return;
        }
        
        try {
            // 确保ONNX环境已初始化
            if (!g_onnxEnv) {
                m_valid = false;
                return;
            }
            
            Ort::SessionOptions options;
            std::wstring w_modelPath = Common::AnsiToWide(modelPath);
            m_session = std::make_unique<Ort::Session>(*g_onnxEnv, w_modelPath.c_str(), options);
            
            // 动态读取模型输入尺寸
            auto inputInfo = m_session->GetInputTypeInfo(0);
            auto tensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
            auto inputShape = tensorInfo.GetShape();
            if (inputShape.size() >= 4) {
                m_inputHeight = static_cast<int>(inputShape[2]);
                m_inputWidth = static_cast<int>(inputShape[3]);
            }
            
            m_valid = true;
            
            // 存储模型路径信息用于会话管理
            m_modelPath = modelPath;
            m_classNamesPath = classNamesPath;
            
        } catch (...) {
            m_valid = false;
        }
    }
    
    bool IsValid() const { return m_valid; }

    int GetInputWidth() const { return m_inputWidth; }
    int GetInputHeight() const { return m_inputHeight; }
    
    std::vector<Detection> RunInference(const char* imagePath) {
        std::vector<Detection> detections;
        
        if (!m_valid || !Common::FileExists(imagePath)) {
            return detections;
        }
        
        try {
            // 加载图像
            cv::Mat image = cv::imread(imagePath);
            if (image.empty()) return detections;
            
            return RunInferenceFromMat(image);
            
        } catch (...) {
            // 推理过程中发生错误
        }
        
        return detections;
    }
    
    // 重载版本：从内存中的cv::Mat对象进行推理
    std::vector<Detection> RunInferenceFromMat(const cv::Mat& image, float confThreshold = 0.25f, float iouThreshold = 0.45f) {
        std::vector<Detection> detections;
        
        if (!m_valid || image.empty()) {
            return detections;
        }
        
        try {
            // 使用 Letterbox 缩放保持长宽比
            cv::Mat letterboxed = LetterboxResize(image, m_inputWidth, m_inputHeight);
            
            // 使用 cv::dnn::blobFromImage 进行高效预处理
            cv::Mat blob = cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, 
                cv::Size(m_inputWidth, m_inputHeight), cv::Scalar(), true, false);
            
            // 获取 blob 数据
            std::vector<float> inputData(blob.begin<float>(), blob.end<float>());
            std::vector<int64_t> inputShape = {1, 3, m_inputHeight, m_inputWidth};
            
            auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, inputData.data(), inputData.size(),
                inputShape.data(), inputShape.size()
            );
            
            // 运行推理
            auto results = m_session->Run(Ort::RunOptions{nullptr}, 
                m_inputNames.data(), &inputTensor, 1,
                m_outputNames.data(), 1);
                
            // 解析 YOLOv11 输出 [1, 84, 8400]
            float* outputData = results[0].GetTensorMutableData<float>();
            auto outputShape = results[0].GetTensorTypeAndShapeInfo().GetShape();
            
            int numClasses = static_cast<int>(outputShape[1]) - 4; // 84 - 4 = 80 类
            int numBoxes = static_cast<int>(outputShape[2]);        // 8400 个框
            
            // 计算缩放比例(用于坐标还原)
            float scaleX = static_cast<float>(image.cols) / m_inputWidth;
            float scaleY = static_cast<float>(image.rows) / m_inputHeight;
            
            // 解析检测结果
            for (int i = 0; i < numBoxes; ++i) {
                // YOLOv11 输出格式: [cx, cy, w, h, class0_prob, class1_prob, ...]
                float cx = outputData[i];
                float cy = outputData[numBoxes + i];
                float w = outputData[2 * numBoxes + i];
                float h = outputData[3 * numBoxes + i];
                
                // 找到最大类别概率
                int maxClassId = 0;
                float maxProb = outputData[4 * numBoxes + i];
                for (int c = 1; c < numClasses; ++c) {
                    float prob = outputData[(4 + c) * numBoxes + i];
                    if (prob > maxProb) {
                        maxProb = prob;
                        maxClassId = c;
                    }
                }
                
                // 置信度过滤
                if (maxProb >= confThreshold) {
                    Detection det;
                    det.confidence = maxProb;
                    
                    // 转换为 x1,y1,x2,y2 格式并还原到原图尺寸
                    det.x = static_cast<int>((cx - w / 2) * scaleX);
                    det.y = static_cast<int>((cy - h / 2) * scaleY);
                    det.width = static_cast<int>(w * scaleX);
                    det.height = static_cast<int>(h * scaleY);
                    
                    if (maxClassId >= 0 && maxClassId < static_cast<int>(m_classNames.size())) {
                        det.className = m_classNames[maxClassId];
                    } else {
                        det.className = "class_" + std::to_string(maxClassId);
                    }
                    
                    detections.push_back(det);
                }
            }
            
            // 应用 NMS
            detections = ApplyNMS(detections, iouThreshold);
            
        } catch (...) {
            // 推理过程中发生错误
        }
        
        return detections;
    }

    // 获取模型路径信息
    const std::string& GetModelPath() const { return m_modelPath; }
    const std::string& GetClassNamesPath() const { return m_classNamesPath; }

private:
    std::unique_ptr<Ort::Session> m_session;
    std::vector<std::string> m_classNames;
    std::vector<const char*> m_inputNames{"images"};
    std::vector<const char*> m_outputNames{"output0"};
    std::string m_modelPath;
    std::string m_classNamesPath;
    int m_inputWidth = 640;   // 模型输入宽度(默认值)
    int m_inputHeight = 640;  // 模型输入高度(默认值)
    bool m_valid = false;
};

// OCR会话类（管理检测、识别和分类三个模型）
class OcrSession {
public:
    OcrSession(const char* detectionModelPath, const char* recognitionModelPath, 
               const char* classifierPath = nullptr, const char* dictPath = nullptr) {
        // 检查文件是否存在
        if (!detectionModelPath || !recognitionModelPath || !Common::FileExists(detectionModelPath) || !Common::FileExists(recognitionModelPath)) {
            m_valid = false;
            return;
        } 
        
        // 分类器是可选的，检查是否提供了路径
        bool useClassifier = classifierPath && strlen(classifierPath) > 0 && Common::FileExists(classifierPath);
        
        try {
            // 初始化ONNX环境（如果尚未初始化）
            if (!g_onnxEnv) {
                g_onnxEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnnxFeature");
            }
            
            // 创建检测模型会话
            Ort::SessionOptions detectionOptions;
            m_detectionSession = std::make_unique<Ort::Session>(*g_onnxEnv, 
                Common::AnsiToWide(detectionModelPath).c_str(), detectionOptions);
            
            // 读取检测模型输入尺寸
            auto inputInfo = m_detectionSession->GetInputTypeInfo(0);
            auto tensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
            auto inputShape = tensorInfo.GetShape();
            if (inputShape.size() >= 4) {
                m_detectionHeight = static_cast<int>(inputShape[2]);
                m_detectionWidth = static_cast<int>(inputShape[3]);
            }
            
            // 创建识别模型会话
            Ort::SessionOptions recognitionOptions;
            m_recognitionSession = std::make_unique<Ort::Session>(*g_onnxEnv, 
                Common::AnsiToWide(recognitionModelPath).c_str(), recognitionOptions);
            
            // 读取识别模型输入尺寸
            auto recInputInfo = m_recognitionSession->GetInputTypeInfo(0);
            auto recTensorInfo = recInputInfo.GetTensorTypeAndShapeInfo();
            auto recInputShape = recTensorInfo.GetShape();
            if (recInputShape.size() >= 4) {
                m_recognitionHeight = static_cast<int>(recInputShape[2]);
                m_recognitionWidth = static_cast<int>(recInputShape[3]);
            }
            
            // 可选：创建分类器模型会话
            if (useClassifier) {
                Ort::SessionOptions classifierOptions;
                m_classifierSession = std::make_unique<Ort::Session>(*g_onnxEnv, 
                    Common::AnsiToWide(classifierPath).c_str(), classifierOptions);
                m_classifierModelPath = classifierPath;
            }
            
            // 加载字典
            m_dictionary = LoadOcrDictionary(dictPath);
            
            m_detectionModelPath = detectionModelPath;
            m_recognitionModelPath = recognitionModelPath;
            m_useClassifier = useClassifier;
            m_valid = true;
            
        } catch (...) {
            m_valid = false;
        }
    }
    
    bool IsValid() const { return m_valid; }
    bool HasClassifier() const { return m_useClassifier; }
    const std::string& GetDetectionModelPath() const { return m_detectionModelPath; }
    const std::string& GetRecognitionModelPath() const { return m_recognitionModelPath; }
    const std::string& GetClassifierModelPath() const { return m_classifierModelPath; }
    
    // OCR文本检测
    std::vector<cv::Rect> DetectTextRegions(const cv::Mat& image, double confidenceThreshold) {
        std::vector<cv::Rect> textRegions;
        
        if (!m_valid) return textRegions;
        
        try {
            // 图像预处理
            cv::Mat processedImage;
            cv::resize(image, processedImage, cv::Size(736, 736));
            processedImage.convertTo(processedImage, CV_32F, 1.0 / 255.0);
            
            // 创建输入张量
            std::vector<int64_t> inputShape = {1, 3, 736, 736};
            std::vector<float> inputData(1 * 3 * 736 * 736);
            
            // HWC到CHW转换并归一化
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < 736; ++h) {
                    for (int w = 0; w < 736; ++w) {
                        inputData[c * 736 * 736 + h * 736 + w] = 
                            processedImage.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, inputData.data(), inputData.size(),
                inputShape.data(), inputShape.size());
            
            // 运行推理
            const char* inputNames[] = {"x"};
            const char* outputNames[] = {"out"};
            auto outputTensors = m_detectionSession->Run(Ort::RunOptions{nullptr}, 
                inputNames, &inputTensor, 1, outputNames, 1);
            
            // 后处理检测结果
            auto* outputData = outputTensors[0].GetTensorData<float>();
            auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            // 简单的后处理：检测文本区域
            // 这里需要根据具体的检测模型调整
            int height = outputShape[2];
            int width = outputShape[3];
            
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = h * width + w;
                    float confidence = outputData[idx];
                    
                    if (confidence > confidenceThreshold) {
                        // 映射回原始图像坐标
                        int origX = w * image.cols / 736;
                        int origY = h * image.rows / 736;
                        int origWidth = 50; // 示例宽度，需要根据模型调整
                        int origHeight = 20; // 示例高度，需要根据模型调整
                        
                        textRegions.push_back(cv::Rect(origX, origY, origWidth, origHeight));
                    }
                }
            }
            
        } catch (...) {
            // 异常处理，返回空结果
        }
        
        return textRegions;
    }
    
    // 文字识别（支持方向分类）
    std::string RecognizeText(const cv::Mat& textRegion) {
        if (!m_valid) return "";
        
        try {
            cv::Mat processedImage = textRegion.clone();
            
            // 如果有分类器，先进行方向分类
            if (m_useClassifier) {
                int angle = ClassifyTextOrientation(processedImage);
                if (angle != 0) {
                    // 旋转图像到正确方向
                    cv::Mat rotated;
                    cv::Point2f center(processedImage.cols/2, processedImage.rows/2);
                    cv::Mat rotMatrix = cv::getRotationMatrix2D(center, -angle, 1.0);
                    cv::warpAffine(processedImage, rotated, rotMatrix, processedImage.size());
                    processedImage = rotated;
                }
            }
            
            // 图像预处理 - 使用动态读取的模型尺寸
            cv::resize(processedImage, processedImage, cv::Size(m_recognitionWidth, m_recognitionHeight));
            
            // 使用 cv::dnn::blobFromImage 进行高效预处理
            cv::Mat blob = cv::dnn::blobFromImage(processedImage, 1.0 / 255.0, 
                cv::Size(m_recognitionWidth, m_recognitionHeight), cv::Scalar(), false, false);
            
            // 获取 blob 数据
            std::vector<float> inputData(blob.begin<float>(), blob.end<float>());
            std::vector<int64_t> inputShape = {1, 3, m_recognitionHeight, m_recognitionWidth};
            
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, inputData.data(), inputData.size(),
                inputShape.data(), inputShape.size());
            
            // 运行推理
            const char* inputNames[] = {"x"};
            const char* outputNames[] = {"softmax_11.tmp_0"};
            auto outputTensors = m_recognitionSession->Run(Ort::RunOptions{nullptr}, 
                inputNames, &inputTensor, 1, outputNames, 1);
            
            // 后处理识别结果 - 使用 CTC 解码
            auto* outputData = outputTensors[0].GetTensorData<float>();
            auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            // 使用 CTC 贪婪解码
            std::string recognizedText = CtcGreedyDecoder(outputData, outputShape, m_dictionary);
            
            return recognizedText;
            
        } catch (...) {
            // 异常处理，返回空字符串
            return "";
        }
    }
    
private:
    // 方向分类
    int ClassifyTextOrientation(const cv::Mat& image) {
        if (!m_valid || !m_useClassifier) return 0;
        
        try {
            // 图像预处理
            cv::Mat processedImage;
            cv::resize(image, processedImage, cv::Size(192, 48));
            processedImage.convertTo(processedImage, CV_32F, 1.0 / 255.0);
            
            // 创建输入张量
            std::vector<int64_t> inputShape = {1, 3, 48, 192};
            std::vector<float> inputData(1 * 3 * 48 * 192);
            
            // HWC到CHW转换
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < 48; ++h) {
                    for (int w = 0; w < 192; ++w) {
                        inputData[c * 48 * 192 + h * 192 + w] = 
                            processedImage.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, inputData.data(), inputData.size(),
                inputShape.data(), inputShape.size());
            
            // 运行推理
            const char* inputNames[] = {"x"};
            const char* outputNames[] = {"out"};
            auto outputTensors = m_classifierSession->Run(Ort::RunOptions{nullptr}, 
                inputNames, &inputTensor, 1, outputNames, 1);
            
            // 后处理分类结果
            auto* outputData = outputTensors[0].GetTensorData<float>();
            
            // 找到最大概率的类别
            int maxIndex = 0;
            float maxProb = outputData[0];
            for (int i = 1; i < 4; ++i) {  // 通常4个方向：0°, 90°, 180°, 270°
                if (outputData[i] > maxProb) {
                    maxProb = outputData[i];
                    maxIndex = i;
                }
            }
            
            // 根据类别返回旋转角度
            switch (maxIndex) {
                case 0: return 0;    // 0度
                case 1: return 90;   // 90度
                case 2: return 180;  // 180度
                case 3: return 270;  // 270度
                default: return 0;
            }
            
        } catch (...) {
            return 0; // 异常时返回0度
        }
    }
    
private:
    std::unique_ptr<Ort::Session> m_detectionSession;
    std::unique_ptr<Ort::Session> m_recognitionSession;
    std::unique_ptr<Ort::Session> m_classifierSession;
    std::string m_detectionModelPath;
    std::string m_recognitionModelPath;
    std::string m_classifierModelPath;
    std::vector<std::string> m_dictionary;  // OCR 字典
    int m_detectionWidth = 736;   // 检测模型输入宽度(默认值)
    int m_detectionHeight = 736;  // 检测模型输入高度(默认值)
    int m_recognitionWidth = 320; // 识别模型输入宽度(默认值)
    int m_recognitionHeight = 48; // 识别模型输入高度(默认值)
    bool m_valid = false;
    bool m_useClassifier = false;
};

// 会话管理辅助函数
static int CreateSession(const char* modelPath, const char* classNamesPath) {
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    
    // 检查是否已存在相同模型的会话
    for (const auto& pair : g_sessions) {
        auto& session = pair.second;
        if (session && session->IsValid() && 
            session->GetModelPath() == modelPath && 
            session->GetClassNamesPath() == classNamesPath) {
            return pair.first; // 返回现有会话ID
        }
    }
    
    // 如果全局环境未初始化，先初始化
    if (!g_onnxEnv) {
        try {
            g_onnxEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnnxFeature");
        } catch (...) {
            return -1; // 初始化失败
        }
    }
    
    // 创建新会话
    auto session = std::make_unique<OnnxSession>(modelPath, classNamesPath);
    if (!session->IsValid()) {
        return -1;
    }
    
    int sessionId = g_nextSessionId++;
    g_sessions[sessionId] = std::move(session);
    return sessionId;
}

static OnnxSession* GetSession(int sessionId) {
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    auto it = g_sessions.find(sessionId);
    if (it != g_sessions.end() && it->second && it->second->IsValid()) {
        return it->second.get();
    }
    return nullptr;
}

// OCR会话管理辅助函数
static int CreateOcrSession(const char* detectionModelPath, const char* recognitionModelPath, const char* classifierPath = nullptr) {
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    
    // 检查是否已存在相同模型的OCR会话
    for (const auto& pair : g_ocrSessions) {
        auto& session = pair.second;
        if (session && session->IsValid() && 
            session->GetDetectionModelPath() == detectionModelPath && 
            session->GetRecognitionModelPath() == recognitionModelPath &&
            session->GetClassifierModelPath() == (classifierPath ? classifierPath : "")) {
            return pair.first; // 返回现有会话ID
        }
    }
    
    // 如果全局环境未初始化，先初始化
    if (!g_onnxEnv) {
        try {
            g_onnxEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnnxFeature");
        } catch (...) {
            return -1; // 初始化失败
        }
    }
    
    // 创建新OCR会话
    auto session = std::make_unique<OcrSession>(detectionModelPath, recognitionModelPath, classifierPath);
    if (!session->IsValid()) {
        return -1;
    }
    
    int sessionId = g_nextOcrSessionId++;
    g_ocrSessions[sessionId] = std::move(session);
    return sessionId;
}

static OcrSession* GetOcrSession(int sessionId) {
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    auto it = g_ocrSessions.find(sessionId);
    if (it != g_ocrSessions.end() && it->second && it->second->IsValid()) {
        return it->second.get();
    }
    return nullptr;
}

// ========== 异步任务管理函数 ==========

// 创建异步任务
static int CreateAsyncTask(const std::string& taskType, int sessionId, std::function<std::string()> taskFunc) {
    std::lock_guard<std::mutex> lock(g_taskMutex);
    
    // 初始化线程池
    InitializeThreadPool();
    
    int taskId = g_nextTaskId++;
    auto task = std::make_unique<AsyncTask>();
    task->taskId = taskId;
    task->status = AsyncTaskStatus::PENDING;
    task->taskType = taskType;
    task->sessionId = sessionId;
    task->startTime = std::chrono::steady_clock::now();
    
    // 更新性能统计
    {
        std::lock_guard<std::mutex> statsLock(g_statsMutex);
        g_performanceStats.totalTasks++;
        g_performanceStats.taskTypeCounts[taskType]++;
    }
    
    // 提交任务到线程池
    auto promisePtr = std::make_shared<std::promise<std::string>>();
    task->future = promisePtr->get_future();
    
    g_threadPool->enqueue([taskId, taskFunc = std::move(taskFunc), promisePtr]() mutable {
        std::string result;
        try {
            // 更新任务状态为运行中
            {
                std::lock_guard<std::mutex> lock(g_taskMutex);
                auto it = g_asyncTasks.find(taskId);
                if (it != g_asyncTasks.end()) {
                    it->second->status = AsyncTaskStatus::RUNNING;
                }
            }
            
            result = taskFunc();
            
            // 更新任务状态为完成
            {
                std::lock_guard<std::mutex> lock(g_taskMutex);
                auto it = g_asyncTasks.find(taskId);
                if (it != g_asyncTasks.end()) {
                    it->second->status = AsyncTaskStatus::COMPLETED;
                    it->second->result = result;
                    
                    // 更新性能统计
                    std::lock_guard<std::mutex> statsLock(g_statsMutex);
                    g_performanceStats.completedTasks++;
                    
                    auto endTime = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - it->second->startTime);
                    g_performanceStats.avgProcessingTimeMs = 
                        (g_performanceStats.avgProcessingTimeMs * (g_performanceStats.completedTasks - 1) + duration.count()) / g_performanceStats.completedTasks;
                }
            }
        } catch (...) {
            result = "-1";
            
            // 更新任务状态为失败
            {
                std::lock_guard<std::mutex> lock(g_taskMutex);
                auto it = g_asyncTasks.find(taskId);
                if (it != g_asyncTasks.end()) {
                    it->second->status = AsyncTaskStatus::FAILED;
                    it->second->result = result;
                    
                    // 更新性能统计
                    std::lock_guard<std::mutex> statsLock(g_statsMutex);
                    g_performanceStats.failedTasks++;
                }
            }
        }
        
        promisePtr->set_value(result);  // ✅ 使用 promisePtr
    });
    
    g_asyncTasks[taskId] = std::move(task);
    return taskId;
}

// 获取异步任务
static AsyncTask* GetAsyncTask(int taskId) {
    std::lock_guard<std::mutex> lock(g_taskMutex);
    auto it = g_asyncTasks.find(taskId);
    if (it != g_asyncTasks.end()) {
        return it->second.get();
    }
    return nullptr;
}

// 清理过期任务（超过1小时）
static void CleanupExpiredTasks() {
    std::lock_guard<std::mutex> lock(g_taskMutex);
    auto now = std::chrono::steady_clock::now();
    
    for (auto it = g_asyncTasks.begin(); it != g_asyncTasks.end();) {
        auto& task = it->second;
        auto duration = std::chrono::duration_cast<std::chrono::hours>(now - task->startTime);
        
        if (duration.count() >= 1) { // 超过1小时
            it = g_asyncTasks.erase(it);
        } else {
            ++it;
        }
    }
}

// ========== OCR流程优化 ==========

// 优化的OCR文本识别函数，减少重复检测和内存拷贝
static std::vector<TextBox> OptimizedOcrRecognition(OcrSession* ocrSession, const cv::Mat& image, double confidenceThreshold) {
    std::vector<TextBox> textBoxes;
    
    if (!ocrSession || image.empty()) {
        return textBoxes;
    }
    
    try {
        // 检测文本区域
        std::vector<cv::Rect> textRegions = ocrSession->DetectTextRegions(image, confidenceThreshold);
        
        // 预处理所有文本区域，减少重复操作
        std::vector<std::pair<cv::Rect, cv::Mat>> preprocessedRegions;
        preprocessedRegions.reserve(textRegions.size());
        
        for (const auto& region : textRegions) {
            // 使用ROI视图，避免不必要的内存拷贝
            cv::Mat textRegion = image(region);
            
            // 只有在需要时才创建拷贝
            if (region.width < 10 || region.height < 10) {
                continue; // 跳过过小的区域
            }
            
            preprocessedRegions.emplace_back(region, textRegion);
        }
        
        // 批量识别文本
        for (const auto& [region, textRegion] : preprocessedRegions) {
            std::string recognizedText = ocrSession->RecognizeText(textRegion);
            
            if (!recognizedText.empty()) {
                TextBox textBox;
                textBox.text = recognizedText;
                textBox.x = region.x;
                textBox.y = region.y;
                textBox.width = region.width;
                textBox.height = region.height;
                textBox.confidence = confidenceThreshold; // 实际应该从检测结果获取
                
                textBoxes.push_back(textBox);
            }
        }
        
    } catch (...) {
        // 异常处理
    }
    
    return textBoxes;
}

// 优化的找字函数
static std::string OptimizedFindText(OcrSession* ocrSession, const cv::Mat& image, const std::string& targetText, 
                                    double confidenceThreshold, int offsetX = 0, int offsetY = 0) {
    if (!ocrSession || image.empty() || targetText.empty()) {
        return "-1";
    }
    
    try {
        // 使用优化的OCR识别
        auto textBoxes = OptimizedOcrRecognition(ocrSession, image, confidenceThreshold);
        
        // 查找目标文字
        for (const auto& textBox : textBoxes) {
            if (textBox.text.find(targetText) != std::string::npos) {
                // 计算文字中心点坐标
                int centerX = textBox.x + offsetX + textBox.width / 2;
                int centerY = textBox.y + offsetY + textBox.height / 2;
                
                return std::to_string(centerX) + "," + std::to_string(centerY);
            }
        }
        
        return "0"; // 未找到
        
    } catch (...) {
        return "-1";
    }
}



// C接口导出
extern "C" {

// ========== 异步API实现 ==========

__declspec(dllexport) int OnnxDetectImageAsync(
    int sessionId,
    const char* imagePath,
    double confidenceThreshold
) {
    if (sessionId <= 0 || !imagePath || !Common::FileExists(imagePath)) {
        return -1;
    }
    
    return CreateAsyncTask("detect_image", sessionId, [sessionId, imagePath = std::string(imagePath), confidenceThreshold]() -> std::string {
        OnnxSession* session = GetSession(sessionId);
        if (!session) return "-1";
        
        // 使用优化的预处理缓存
        auto preprocessed = ImagePreprocessor::PreprocessAndCache(imagePath, session->GetInputWidth(), session->GetInputHeight());
        if (preprocessed.image.empty()) return "-1";
        
        auto detections = session->RunInferenceFromMat(preprocessed.image, confidenceThreshold);
        
        if (detections.empty()) return "0";
        
        std::stringstream result;
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& det = detections[i];
            if (det.confidence >= confidenceThreshold) {
                if (i > 0) result << "|";
                result << det.className << "," 
                      << det.confidence << "," 
                      << det.x << "," 
                      << det.y << "," 
                      << det.width << "," 
                      << det.height;
            }
        }
        
        std::string resultStr = result.str();
        return resultStr.empty() ? "0" : resultStr;
    });
}

__declspec(dllexport) int OnnxDetectWindowAsync(
    int sessionId,
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    double confidenceThreshold
) {
    if (sessionId <= 0 || !IsWindow(hwnd) || roiWidth <= 0 || roiHeight <= 0) {
        return -1;
    }
    
    return CreateAsyncTask("detect_window", sessionId, [sessionId, hwnd, roiX, roiY, roiWidth, roiHeight, confidenceThreshold]() -> std::string {
        OnnxSession* session = GetSession(sessionId);
        if (!session) return "-1";
        
        cv::Mat screenshot;
        if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, screenshot)) {
            return "-1";
        }
        
        auto detections = session->RunInferenceFromMat(screenshot, confidenceThreshold);
        
        if (detections.empty()) return "0";
        
        std::stringstream result;
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& det = detections[i];
            if (det.confidence >= confidenceThreshold) {
                if (i > 0) result << "|";
                result << det.className << "," 
                      << det.confidence << "," 
                      << (det.x + roiX) << "," 
                      << (det.y + roiY) << "," 
                      << det.width << "," 
                      << det.height;
            }
        }
        
        std::string resultStr = result.str();
        return resultStr.empty() ? "0" : resultStr;
    });
}

__declspec(dllexport) int OnnxOcrImageAsync(
    int ocrSessionId,
    const char* imagePath,
    double confidenceThreshold
) {
    if (ocrSessionId < 10000 || !imagePath || !Common::FileExists(imagePath)) {
        return -1;
    }
    
    return CreateAsyncTask("ocr_image", ocrSessionId, [ocrSessionId, imagePath = std::string(imagePath), confidenceThreshold]() -> std::string {
        OcrSession* ocrSession = GetOcrSession(ocrSessionId);
        if (!ocrSession) return "-1";
        
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) return "-1";
        
        // 使用优化的OCR识别
        auto textBoxes = OptimizedOcrRecognition(ocrSession, image, confidenceThreshold);
        
        if (textBoxes.empty()) return "";
        
        std::stringstream result;
        bool firstText = true;
        
        for (const auto& textBox : textBoxes) {
            if (!firstText) result << "|";
            result << textBox.text << "," << textBox.x << "," << textBox.y << "," 
                   << textBox.width << "," << textBox.height;
            firstText = false;
        }
        
        return result.str();
    });
}

__declspec(dllexport) int OnnxOcrWindowAsync(
    int ocrSessionId,
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    double confidenceThreshold
) {
    if (ocrSessionId < 10000 || !IsWindow(hwnd) || roiWidth <= 0 || roiHeight <= 0) {
        return -1;
    }
    
    return CreateAsyncTask("ocr_window", ocrSessionId, [ocrSessionId, hwnd, roiX, roiY, roiWidth, roiHeight, confidenceThreshold]() -> std::string {
        OcrSession* ocrSession = GetOcrSession(ocrSessionId);
        if (!ocrSession) return "-1";
        
        cv::Mat screenshot;
        if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, screenshot)) {
            return "-1";
        }
        
        // 使用优化的OCR识别
        auto textBoxes = OptimizedOcrRecognition(ocrSession, screenshot, confidenceThreshold);
        
        if (textBoxes.empty()) return "";
        
        std::stringstream result;
        bool firstText = true;
        
        for (const auto& textBox : textBoxes) {
            if (!firstText) result << "|";
            result << textBox.text << "," << (textBox.x + roiX) << "," << (textBox.y + roiY) << "," 
                   << textBox.width << "," << textBox.height;
            firstText = false;
        }
        
        return result.str();
    });
}

__declspec(dllexport) const char* OnnxGetAsyncResult(
    int taskId,
    int timeoutMs
) {
    // 清理过期任务
    CleanupExpiredTasks();
    
    AsyncTask* task = GetAsyncTask(taskId);
    if (!task) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    // 检查任务状态
    if (task->status == AsyncTaskStatus::COMPLETED || task->status == AsyncTaskStatus::FAILED) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), task->result.c_str());
        return g_resultBuffer;
    }
    
    if (task->status == AsyncTaskStatus::CANCELLED) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    // 等待任务完成
    if (timeoutMs == 0) {
        // 立即返回
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "");
        return g_resultBuffer;
    }
    
    auto status = task->future.wait_for(std::chrono::milliseconds(timeoutMs));
    
    if (status == std::future_status::ready) {
        try {
            std::string result = task->future.get();
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), result.c_str());
            return g_resultBuffer;
        } catch (...) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
            return g_resultBuffer;
        }
    } else if (status == std::future_status::timeout) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "timeout");
        return g_resultBuffer;
    }
    
    strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    return g_resultBuffer;
}

__declspec(dllexport) int OnnxCancelAsyncTask(int taskId) {
    AsyncTask* task = GetAsyncTask(taskId);
    if (!task) return -1;
    
    if (task->status == AsyncTaskStatus::RUNNING || task->status == AsyncTaskStatus::PENDING) {
        task->status = AsyncTaskStatus::CANCELLED;
        
        // 更新性能统计
        {
            std::lock_guard<std::mutex> statsLock(g_statsMutex);
            g_performanceStats.cancelledTasks++;
        }
        
        return 1;
    }
    
    return -1;
}

__declspec(dllexport) const char* OnnxGetPerformanceStats(int sessionId) {
    std::lock_guard<std::mutex> lock(g_statsMutex);
    
    std::stringstream json;
    json << "{\n";
    json << "  \"totalTasks\": " << g_performanceStats.totalTasks << ",\n";
    json << "  \"completedTasks\": " << g_performanceStats.completedTasks << ",\n";
    json << "  \"failedTasks\": " << g_performanceStats.failedTasks << ",\n";
    json << "  \"cancelledTasks\": " << g_performanceStats.cancelledTasks << ",\n";
    json << "  \"avgProcessingTimeMs\": " << g_performanceStats.avgProcessingTimeMs << ",\n";
    
    json << "  \"taskTypeCounts\": {\n";
    bool first = true;
    for (const auto& [type, count] : g_performanceStats.taskTypeCounts) {
        if (!first) json << ",\n";
        json << "    \"" << type << "\": " << count;
        first = false;
    }
    json << "\n  }\n";
    json << "}";
    
    std::string result = json.str();
    if (result.length() < sizeof(g_resultBuffer)) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), result.c_str());
    } else {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "{}");
    }
    
    return g_resultBuffer;
}

__declspec(dllexport) int OnnxSetThreadPoolSize(int numThreads) {
    std::lock_guard<std::mutex> lock(g_threadPoolMutex);
    
    if (numThreads < 0) return -1;
    
    g_threadPoolSize = numThreads;
    
    // 重新初始化线程池
    if (g_threadPool) {
        g_threadPool.reset();
    }
    
    InitializeThreadPool();
    return 1;
}

/**
 * @brief 创建ONNX会话并加载模型
 * @param modelPath ONNX模型文件路径
 * @param classNamesPath 类名文件路径
 * @return 会话ID，成功返回>0，失败返回-1
 */
__declspec(dllexport) int OnnxCreateSession(
    const char* modelPath,     // ONNX模型文件路径
    const char* classNamesPath  // 类名文件路径
) {
    if (!modelPath || !classNamesPath) {
        return -1;
    }
    
    if (!Common::FileExists(modelPath) || !Common::FileExists(classNamesPath)) {
        return -1;
    }
    
    return CreateSession(modelPath, classNamesPath);
}

/**
 * @brief 创建ONNX OCR会话并加载模型
 * @param detectionModelPath OCR文本检测ONNX模型文件路径（必填）
 * @param recognitionModelPath OCR文字识别ONNX模型文件路径（必填）
 * @param classifierPath OCR方向分类器ONNX模型文件路径（可选，传入NULL或空字符串表示不使用）
 * @return OCR会话ID，成功返回>10000，失败返回-1
 */
__declspec(dllexport) int OnnxCreateOcrSession(
    const char* detectionModelPath,  // OCR文本检测模型路径
    const char* recognitionModelPath, // OCR文字识别模型路径
    const char* classifierPath       // OCR方向分类器模型路径（可选）
) {
    if (!detectionModelPath || !recognitionModelPath) {
        return -1;
    }
    
    if (!Common::FileExists(detectionModelPath) || !Common::FileExists(recognitionModelPath)) {
        return -1;
    }
    
    return CreateOcrSession(detectionModelPath, recognitionModelPath, classifierPath);
}






/**
 * @brief 使用会话ID对图像文件进行ONNX目标检测推理
 * @param sessionId 会话ID（由OnnxCreateSession返回）
 * @param imagePath 待检测图像文件路径
 * @param confidenceThreshold 置信度阈值 (0.0 - 1.0)
 * @return 检测结果字符串，格式为："类别名,置信度,x,y,宽度,高度|..."，
 *         未检测到目标返回"0"，发生错误返回"-1"
 */
__declspec(dllexport) const char* OnnxDetectImage(
    int sessionId,            // 会话ID
    const char* imagePath,     // 待检测图像路径
    double confidenceThreshold // 置信度阈值
) {
    // 初始化返回缓冲区
    strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    
    // 参数检查
    if (sessionId <= 0 || !imagePath) {
        return g_resultBuffer;
    }
    
    // 检查文件是否存在
    if (!Common::FileExists(imagePath)) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    // 获取会话
    OnnxSession* session = GetSession(sessionId);
    if (!session) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    try {
        // 运行推理
        auto detections = session->RunInference(imagePath);
        
        if (detections.empty()) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "0");
            return g_resultBuffer;
        }
        
        // 格式化结果字符串
        std::stringstream result;
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& det = detections[i];
            if (det.confidence >= confidenceThreshold) {
                if (i > 0) result << "|";
                result << det.className << "," 
                      << det.confidence << "," 
                      << det.x << "," 
                      << det.y << "," 
                      << det.width << "," 
                      << det.height;
            }
        }
        
        std::string resultStr = result.str();
        if (resultStr.empty()) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "0");
        } else {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), resultStr.c_str());
        }
        
    } catch (...) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    }
    
    return g_resultBuffer;
}



// 清理所有ONNX相关资源
__declspec(dllexport) void OnnxPlugin_Cleanup() {
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    
    // 先释放所有会话
    g_sessions.clear();
    
    // 释放所有OCR会话
    g_ocrSessions.clear();
    
    // 再清理全局ONNX环境
    if (g_onnxEnv) {
        g_onnxEnv.reset();
        g_onnxEnv = nullptr;
    }
}

/**
 * @brief 使用OCR会话ID对图像文件进行OCR文字识别
 * @param ocrSessionId OCR会话ID（由OnnxCreateOcrSession返回）
 * @param imagePath 待识别的图像文件路径
 * @param confidenceThreshold 置信度阈值 (0.0 - 1.0)
 * @return 识别结果字符串，格式为："文本1,x1,y1,w1,h1|文本2,x2,y2,w2,h2|..."，
 *         未识别到文本返回""，发生错误返回"-1"
 */
__declspec(dllexport) const char* OnnxOcrImage(
    int ocrSessionId,        // OCR会话ID
    const char* imagePath,    // 待识别图像路径
    double confidenceThreshold // 置信度阈值
) {
    // 初始化返回缓冲区
    strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    
    // 参数检查
    if (ocrSessionId < 10000 || !imagePath) {
        return g_resultBuffer;
    }
    
    // 检查文件是否存在
    if (!Common::FileExists(imagePath)) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    // 获取OCR会话
    OcrSession* ocrSession = GetOcrSession(ocrSessionId);
    if (!ocrSession) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    try {
        // 加载图像
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
            return g_resultBuffer;
        }
        
        // 使用优化的OCR识别
        auto textBoxes = OptimizedOcrRecognition(ocrSession, image, confidenceThreshold);
        
        if (textBoxes.empty()) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "");
            return g_resultBuffer;
        }
        
        // 构建结果字符串
        std::stringstream result;
        bool firstText = true;
        
        for (const auto& textBox : textBoxes) {
            if (!firstText) result << "|";
            result << textBox.text << "," << textBox.x << "," << textBox.y << "," 
                   << textBox.width << "," << textBox.height;
            firstText = false;
        }
        
        std::string resultStr = result.str();
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), resultStr.c_str());
        
    } catch (...) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    }
    
    return g_resultBuffer;
}

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
__declspec(dllexport) const char* OnnxOcrWindow(
    int ocrSessionId,        // OCR会话ID
    HWND hwnd,               // 目标窗口句柄
    int roiX,                // 截图区域X坐标
    int roiY,                // 截图区域Y坐标
    int roiWidth,            // 截图区域宽度
    int roiHeight,           // 截图区域高度
    double confidenceThreshold // 置信度阈值
) {
    // 初始化返回缓冲区
    strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    
    // 参数检查
    if (ocrSessionId < 10000 || !IsWindow(hwnd) || roiWidth <= 0 || roiHeight <= 0) {
        return g_resultBuffer;
    }
    
    // 获取OCR会话
    OcrSession* ocrSession = GetOcrSession(ocrSessionId);
    if (!ocrSession) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    try {
        // 截取窗口区域
        cv::Mat screenshot;
        if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, screenshot)) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
            return g_resultBuffer;
        }
        
        // 使用优化的OCR识别
        auto textBoxes = OptimizedOcrRecognition(ocrSession, screenshot, confidenceThreshold);
        
        if (textBoxes.empty()) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "");
            return g_resultBuffer;
        }
        
        // 构建结果字符串
        std::stringstream result;
        bool firstText = true;
        
        for (const auto& textBox : textBoxes) {
            if (!firstText) result << "|";
            result << textBox.text << "," << (textBox.x + roiX) << "," << (textBox.y + roiY) << "," 
                   << textBox.width << "," << textBox.height;
            firstText = false;
        }
        
        std::string resultStr = result.str();
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), resultStr.c_str());
        
    } catch (...) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    }
    
    return g_resultBuffer;
}

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
__declspec(dllexport) const char* PaddleOcrFindText(
    int ocrSessionId,        // OCR会话ID
    HWND hwnd,               // 目标窗口句柄
    int roiX,                // 查找区域X坐标
    int roiY,                // 查找区域Y坐标
    int roiWidth,            // 查找区域宽度
    int roiHeight,           // 查找区域高度
    const char* targetText,  // 要查找的目标文字
    double confidenceThreshold // 置信度阈值
) {
    // 初始化返回缓冲区
    strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    
    // 参数检查
    if (ocrSessionId < 10000 || !IsWindow(hwnd) || roiWidth <= 0 || roiHeight <= 0 || !targetText || !*targetText) {
        return g_resultBuffer;
    }
    
    // 获取OCR会话
    OcrSession* ocrSession = GetOcrSession(ocrSessionId);
    if (!ocrSession) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    try {
        // 截取窗口区域
        cv::Mat screenshot;
        if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, screenshot)) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
            return g_resultBuffer;
        }
        
        // 使用优化的找字函数
        std::string result = OptimizedFindText(ocrSession, screenshot, targetText, confidenceThreshold, roiX, roiY);
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), result.c_str());
        
    } catch (...) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    }
    
    return g_resultBuffer;
}

/**
 * @brief PaddleOCR找字API - 在图像文件中查找特定文字
 * @param ocrSessionId OCR会话ID（由OnnxCreateOcrSession返回）
 * @param imagePath 图像文件路径
 * @param targetText 要查找的目标文字
 * @param confidenceThreshold 置信度阈值 (0.0 - 1.0)
 * @return 找到文字返回"x,y"坐标字符串，未找到返回"0"，发生错误返回"-1"
 */
__declspec(dllexport) const char* PaddleOcrFindTextInImage(
    int ocrSessionId,        // OCR会话ID
    const char* imagePath,    // 图像文件路径
    const char* targetText,   // 要查找的目标文字
    double confidenceThreshold // 置信度阈值
) {
    // 初始化返回缓冲区
    strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    
    // 参数检查
    if (ocrSessionId < 10000 || !imagePath || !targetText || !*targetText) {
        return g_resultBuffer;
    }
    
    // 检查文件是否存在
    if (!Common::FileExists(imagePath)) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    // 获取OCR会话
    OcrSession* ocrSession = GetOcrSession(ocrSessionId);
    if (!ocrSession) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    try {
        // 加载图像
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
            return g_resultBuffer;
        }
        
        // 使用优化的找字函数
        std::string result = OptimizedFindText(ocrSession, image, targetText, confidenceThreshold);
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), result.c_str());
        
    } catch (...) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    }
    
    return g_resultBuffer;
}

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
__declspec(dllexport) const char* OnnxDetectWindow(
    int sessionId,           // 会话ID
    HWND hwnd,               // 目标窗口句柄
    int roiX,                // 截图区域X坐标
    int roiY,                // 截图区域Y坐标
    int roiWidth,            // 截图区域宽度
    int roiHeight,           // 截图区域高度
    double confidenceThreshold // 置信度阈值
) {
    // 初始化返回缓冲区
    strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    
    // 参数检查
    if (sessionId <= 0 || !IsWindow(hwnd) || roiWidth <= 0 || roiHeight <= 0) {
        return g_resultBuffer;
    }
    
    // 获取会话
    OnnxSession* session = GetSession(sessionId);
    if (!session) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
        return g_resultBuffer;
    }
    
    try {
        // 截取窗口区域
        cv::Mat screenshot;
        if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, screenshot)) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
            return g_resultBuffer;
        }
        
        // 运行推理
        auto detections = session->RunInferenceFromMat(screenshot);
        
        if (detections.empty()) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "0");
            return g_resultBuffer;
        }
        
        // 格式化结果字符串
        std::stringstream result;
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& det = detections[i];
            if (det.confidence >= confidenceThreshold) {
                if (i > 0) result << "|";
                result << det.className << "," 
                      << det.confidence << "," 
                      << (det.x + roiX) << ","  // 坐标转换为相对于窗口的绝对坐标
                      << (det.y + roiY) << "," 
                      << det.width << "," 
                      << det.height;
            }
        }
        
        std::string resultStr = result.str();
        if (resultStr.empty()) {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "0");
        } else {
            strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), resultStr.c_str());
        }
        
    } catch (...) {
        strcpy_s(g_resultBuffer, sizeof(g_resultBuffer), "-1");
    }
    
    return g_resultBuffer;
}





}
