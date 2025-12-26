#include "HHYolo.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <new> // for std::nothrow
#include <string>
#include <sstream>
#include <tlhelp32.h>
#include <algorithm>
#include <memory>
#include <cctype>
#include <filesystem>
#include <tesseract/baseapi.h>
#include <windows.h> // MessageBoxA
#include <winternl.h> // for NTSTATUS
#include "../Common/Utils.h"

// NTSTATUS 定义（如果 winternl.h 没有提供）
#ifndef NT_SUCCESS
#define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)
#endif



namespace {

} // namespace

// --- 编码转换辅助函数 ---
// 已移动到 Common/Utils.h

static std::wstring ResolveModelPath(
    const char* modelPath,
    const wchar_t* defaultFileName)
{
    if (modelPath != nullptr && modelPath[0] != '\0') {
        return Common::AnsiToWide(std::string(modelPath));
    }

    wchar_t modulePathBuffer[MAX_PATH] = {};
    if (GetModuleFileNameW(nullptr, modulePathBuffer, MAX_PATH) == 0) {
        return std::wstring();
    }

    std::filesystem::path modulePath(modulePathBuffer);
    modulePath = modulePath.parent_path() / defaultFileName;
    return modulePath.native();
}

// =================================================================================
// OCR Core Logic Refactoring
// =================================================================================

// 用于保存OCR识别结果的内部结构体
struct OcrWord {
    std::string text_utf8; // 识别出的文本 (UTF-8编码)
    int x;                 // 文本在原始输入图像中的坐标和尺寸
    int y;
    int width;
    int height;
};

/**
 * @brief 内部核心OCR函数，封装了图像预处理和Tesseract识别的完整流程。
 * @param inputImage 输入的图像 (BGR或BGRA格式)。
 * @param lang Tesseract语言参数。
 * @param tessdataPath Tesseract数据路径。
 * @return 包含所有识别出的单词及其坐标的向量。
 */
static std::vector<OcrWord> InternalOcr(
    const cv::Mat& inputImage,
    const char* lang,
    const char* tessdataPath
) {
    std::vector<OcrWord> results; // 用于存储最终结果的向量
    if (inputImage.empty()) {
        return results; // 如果输入图像为空，直接返回空结果
    }

    // --- 1. 图像预处理 ---
    cv::Mat processedImage; // 用于存储最终送入Tesseract的图像
    const float scale = 2.0f; // 放大倍数，2倍对屏幕截图足够
    const int border = Common::OCR_BORDER_SIZE; // 添加到图像周围的白色边框大小

    try {
        // --- 1.1. 转换为灰度图 ---
        cv::Mat grayMat;
        if (inputImage.channels() == 4) {
            cv::cvtColor(inputImage, grayMat, cv::COLOR_BGRA2GRAY);
        } else if (inputImage.channels() == 3) {
            cv::cvtColor(inputImage, grayMat, cv::COLOR_BGR2GRAY);
        } else {
            grayMat = inputImage.clone();
        }

        // --- 1.2. 图像缩放 ---
        // 使用三次样条插值放大图像
        cv::resize(grayMat, grayMat, cv::Size(), scale, scale, cv::INTER_CUBIC);

        // --- 1.3. 对比度增强（仅对低对比度图像） ---
        cv::Scalar m, s;
        cv::meanStdDev(grayMat, m, s);
        if (s[0] < Common::OCR_LOW_CONTRAST_THRESHOLD) {
            // 低对比度图像才使用 CLAHE 增强
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            clahe->apply(grayMat, grayMat);
        }

        // --- 1.4. 动态二值化 ---
        cv::Mat bin;
        cv::meanStdDev(grayMat, m, s); // 重新计算（可能经过CLAHE处理）
        if (s[0] < Common::OCR_LOW_CONTRAST_THRESHOLD) {
            // 低对比度：使用自适应阈值
            cv::adaptiveThreshold(grayMat, bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
        } else {
            // 高对比度：使用 OTSU 全局阈值
            cv::threshold(grayMat, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        }

        // --- 1.5. 确保白底黑字 ---
        // Tesseract 在白底黑字上表现更好
        if (cv::mean(bin)[0] < 127) {
            cv::bitwise_not(bin, bin);
        }

        // --- 1.6. 轻微去噪（二值化后） ---
        // 使用小核开运算去除孤立噪点，不影响文字
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel);
        
        // --- 1.7. 添加边框 ---
        cv::copyMakeBorder(bin, processedImage, border, border, border, border, cv::BORDER_CONSTANT, cv::Scalar(255));
    
    } catch (...) {
        return results;
    }

    // --- 2. Tesseract OCR (使用 thread_local 缓存实例) ---
    static thread_local std::unique_ptr<tesseract::TessBaseAPI> api;
    static thread_local std::string currentLang;
    static thread_local std::string currentTessdata;

    if (!api) {
        api.reset(new tesseract::TessBaseAPI());
    }

    const char* langToUse = (lang && *lang) ? lang : "chi_sim";
    const char* tessdataToUse = tessdataPath ? tessdataPath : "";
    
    // 检查是否需要重新初始化 (语言或数据路径改变)
    if (currentLang != langToUse || currentTessdata != tessdataToUse) {
        api->End();
        
        bool initSuccess = false;
        // 优先使用用户提供的路径
        if (tessdataPath && *tessdataPath && api->Init(tessdataPath, langToUse, tesseract::OEM_LSTM_ONLY) == 0) {
            initSuccess = true;
        } else {
            // 其次尝试 "path/tessdata"
            if (tessdataPath && *tessdataPath) {
                std::string altPath = std::string(tessdataPath) + "/tessdata";
                if (api->Init(altPath.c_str(), langToUse, tesseract::OEM_LSTM_ONLY) == 0) {
                    initSuccess = true;
                }
            }
            // 最后尝试系统路径
            if (!initSuccess && api->Init(NULL, langToUse, tesseract::OEM_LSTM_ONLY) == 0) {
                initSuccess = true;
            }
        }

        if (!initSuccess) {
            return results; // 初始化失败
        }

        currentLang = langToUse;
        currentTessdata = tessdataToUse;

        // --- 2.2. 设置Tesseract参数 ---
        api->SetSourceResolution(300);
        // PSM_SINGLE_BLOCK (6): 假设图像是单个统一的文本块
        api->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
        api->SetVariable("preserve_interword_spaces", "1");
    }

    try {
        // --- 2.3. 执行识别 ---
        api->SetImage(processedImage.data, processedImage.cols, processedImage.rows, 1, processedImage.step);
        api->Recognize(nullptr);

        // --- 2.4. 遍历识别结果 ---
        tesseract::ResultIterator* it = api->GetIterator();
        if (it) {
            do {
                int left, top, right, bottom;
                if (it->BoundingBox(tesseract::RIL_WORD, &left, &top, &right, &bottom)) {
                    std::unique_ptr<char[]> word_ptr(it->GetUTF8Text(tesseract::RIL_WORD));
                    if (word_ptr) {
                        OcrWord ocrWord;
                        ocrWord.text_utf8 = std::string(word_ptr.get());
                        
                        // --- 2.5. 坐标转换 ---
                        ocrWord.x = static_cast<int>((left - border) / scale);
                        ocrWord.y = static_cast<int>((top - border) / scale);
                        ocrWord.width = static_cast<int>((right - left) / scale);
                        ocrWord.height = static_cast<int>((bottom - top) / scale);

                        results.push_back(ocrWord);
                    }
                }
            } while (it->Next(tesseract::RIL_WORD));
            delete it;
        }
    } catch (...) {
        // 捕获未知异常
    }

    // 不再删除 api，也不调用 End()，以便复用
    // api->End(); 
    // delete api;

    return results; // 返回所有识别出的单词信息
}

/**
 * @brief 在一个图像文件中查找一个模板图片的位置。
 * 
 * @param imgPath 主图像的文件路径。
 * @param tplPath 模板图片的文件路径。
 * @param similarity 相似度阈值 (0.0 - 1.0)。匹配结果的相似度必须高于此值才被认为是有效的。
 * @param x [out] 如果找到，此指针指向的变量将被设为模板左上角在主图中的X坐标。
 * @param y [out] 如果找到，此指针指向的变量将被设为模板左上角在主图中的Y坐标。
 * @param confidence [out] 如果找到，此指针指向的变量将被设为实际的匹配相似度。
 * 
 * @return int - 1: 成功找到。
 *             - 0: 未找到（相似度低于阈值或图片加载失败）。
 *             - -1: 发生未知异常。
 */
int __stdcall FindImage(
    const char* imgPath,
    const char* tplPath,
    double similarity,
    int* x,
    int* y,
    double* confidence)
{
    // --- 1. 参数校验和初始化 ---
    if (!imgPath || !tplPath || !x || !y || !confidence) return 0; // 检查所有指针参数是否为空，如果任意一个为空，则无法继续执行，直接返回0表示失败。
    *x = *y = -1; // 初始化输出参数，将坐标设为-1，作为未找到的默认值。
    *confidence = 0.0; // 初始化置信度为0.0。

    try { // 使用 try-catch 块来捕获 OpenCV 可能抛出的任何异常，增强程序的健壮性。
        // --- 2. 加载图像 ---
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR); // 从指定路径加载主图像，以彩色模式读取。
        cv::Mat tpl = cv::imread(tplPath, cv::IMREAD_COLOR); // 从指定路径加载模板图像，以彩色模式读取。
        if (img.empty() || tpl.empty()) { // 检查主图或模板图是否加载失败（例如，路径错误或文件损坏）。
            return 0; // 如果任一图像为空，则返回0表示未找到。
        }

        // --- 3. 执行模板匹配 ---
        cv::Mat result; // 创建一个 Mat 对象来存储匹配结果。
        // 调用 OpenCV 的 matchTemplate 函数进行模板匹配。
        // 使用 TM_CCOEFF_NORMED 方法，它计算归一化的相关系数，结果越接近1.0表示匹配度越高。
        cv::matchTemplate(img, tpl, result, cv::TM_CCOEFF_NORMED);

        // --- 4. 查找最佳匹配位置 ---
        double minVal, maxVal; // 定义变量来存储匹配结果中的最小值、最大值。
        cv::Point minLoc, maxLoc; // 定义变量来存储匹配结果中的最小值、最大值对应的位置。
        // 在结果矩阵中查找全局最小值和最大值及其位置。对于 TM_CCOEFF_NORMED 方法，我们关心的是最大值。
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        // --- 5. 检查相似度并返回结果 ---
        if (maxVal >= similarity) { // 检查找到的最佳匹配的相似度（maxVal）是否大于或等于用户设定的阈值。
            *x = maxLoc.x; // 如果满足条件，则将最佳匹配位置的左上角X坐标（maxLoc.x）赋值给输出参数 x。
            *y = maxLoc.y; // 将最佳匹配位置的左上角Y坐标（maxLoc.y）赋值给输出参数 y。
            *confidence = maxVal; // 将实际的最高相似度赋值给输出参数 confidence。
            return 1; // 返回 1 表示成功找到匹配。
        }
    } catch (...) {
        return -1; // 如果在 try 块中的任何位置发生异常，捕获它并返回 -1 表示发生了错误。
    }
    return 0; // 如果代码执行到这里，意味着 try 块正常结束但没有找到满足相似度条件的匹配。
}

/**
 * @brief 截取指定窗口的特定区域(ROI)，并在该区域内查找单个模板图片。
 * 
 * @param hwnd 目标窗口的句柄。
 * @param roiX, roiY, roiWidth, roiHeight 定义在窗口客户区内进行查找的矩形区域 (Region of Interest)。
 * @param tplPath 模板图片的文件路径。
 * @param similarity 相似度阈值 (0.0 - 1.0)。
 * @param matchX [out] 如果找到，返回匹配位置左上角相对于【窗口】的X坐标。
 * @param matchY [out] 如果找到，返回匹配位置左上角相对于【窗口】的Y坐标。
 * @param confidence [out] 如果找到，返回实际的匹配相似度。
 * 
 * @return int - 1: 成功找到。
 *             - 0: 未找到。
 *             - -1: 内部找图函数发生异常。
 *             - -2: 输入参数无效 (如句柄、路径为空，ROI尺寸非法)。
 *             - -3: GDI资源创建失败。
 *             - -4: BitBlt 截图失败。
 *             - -5: 内存分配失败。
 */
int __stdcall CaptureAndFindImage(
    HWND hwnd,
    int roiX, int roiY, int roiWidth, int roiHeight,
    const char* tplPath,
    double similarity,
    int* matchX,
    int* matchY,
    double* confidence
) {
    // --- 1. 参数校验和初始化 ---
    if (!IsWindow(hwnd) || !tplPath || !matchX || !matchY || !confidence || roiWidth <= 0 || roiHeight <= 0) { // 检查窗口句柄是否有效，模板路径和输出参数指针是否为空，以及ROI尺寸是否合法。
        return -2; // 如果参数无效，返回错误码 -2。
    }
    *matchX = -1; // 初始化输出参数，将匹配坐标设为-1，表示未找到。
    *matchY = -1;
    *confidence = 0.0; // 初始化置信度为0.0。

    // --- 2. 截图 (使用 Common 工具) ---
    cv::Mat screenMat;
    if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, screenMat)) {
        return -3;
    }

    // --- 3. 调用核心内存找图函数 ---
    int foundX = -1, foundY = -1;
    double foundConfidence = 0.0;
    
    // FindImageFromMem 支持 3 通道 (BGR) 或 4 通道 (BGRA)
    // Common::CaptureWindowRegion 返回 BGR (3通道)
    int result = FindImageFromMem(
        screenMat.data,  // 图像数据
        roiWidth,        // 宽度
        roiHeight,       // 高度
        3,               // 通道数 (BGR)
        tplPath,         // 模板路径
        similarity,      // 相似度
        &foundX,         // 输出X
        &foundY,         // 输出Y
        &foundConfidence // 输出置信度
    );

    // bmpBuffer 不再需要手动释放，screenMat 会自动管理内存

    // --- 6. 坐标转换并返回 ---
    if (result == 1) { // 如果内存找图成功（返回值为1）。
        // FindImageFromMem 返回的坐标是相对于ROI的，需要加上ROI的起始坐标，转换成相对于整个窗口的坐标。
        *matchX = roiX + foundX;
        *matchY = roiY + foundY;
        *confidence = foundConfidence; // 将找到的实际置信度赋值给输出参数。
    }

    return result; // 返回内存找图函数的结果。
}

/**
 * @brief 截取指定窗口的特定区域(ROI)，并在该区域内查找一个模板图片的所有匹配项。
 * 
 * @param hwnd 目标窗口的句柄。
 * @param roiX, roiY, roiWidth, roiHeight 定义在窗口客户区内进行查找的矩形区域。
 * @param tplPath 模板图片的文件路径。
 * @param similarity 相似度阈值 (0.0 - 1.0)。
 * 
 * @return const char* 一个字符串，格式为 "数量;x1,y1,sim1|x2,y2,sim2|..."
 *         - 数量: 找到的匹配总数。
 *         - x,y: 每个匹配项左上角相对于【窗口】的坐标。
 *         - sim: 每个匹配项的实际相似度。
 *         - 如果未找到，返回 "0;"。
 *         - 如果发生错误，返回 "-1"。
 */
const char* __stdcall CaptureAndFindAllImages(
    HWND hwnd,
    int roiX, int roiY, int roiWidth, int roiHeight,
    const char* tplPath,
    double similarity
) {
    thread_local char resultBuffer[2048]; // 使用 thread_local 替代 static，确保线程安全
    const char* errorResult = "-1"; // 定义一个表示错误的默认返回字符串。
    strcpy_s(resultBuffer, sizeof(resultBuffer), errorResult); // 首先将结果缓冲区初始化为错误字符串，这样在任何错误路径下都会返回 "-1"。

    // --- 1. 参数校验 ---
    if (!IsWindow(hwnd) || !tplPath || roiWidth <= 0 || roiHeight <= 0) { // 检查窗口句柄、模板路径和ROI尺寸的有效性。
        return resultBuffer; // 如果参数无效，直接返回默认的错误字符串。
    }

    // --- 2. 截图 (使用 Common 工具) ---
    cv::Mat imgToMatch;
    if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, imgToMatch)) {
        return resultBuffer;
    }

    // --- 4. 循环查找所有匹配项 ---
    try {
        // --- 4.1. 准备模板图 ---
        cv::Mat tpl = cv::imread(tplPath, cv::IMREAD_COLOR);
        if (tpl.empty()) {
            strcpy_s(resultBuffer, sizeof(resultBuffer), "0;");
            return resultBuffer;
        }
        // imgToMatch 已经是 BGR 格式，无需转换

        // --- 4.2. 执行一次模板匹配，得到包含所有潜在匹配位置的结果矩阵 ---
        cv::Mat result;
        cv::matchTemplate(imgToMatch, tpl, result, cv::TM_CCOEFF_NORMED);

        std::stringstream coords_ss; // 使用 stringstream 来高效地拼接坐标字符串。
        int match_count = 0; // 初始化匹配计数器为0。
        
        // --- 4.3. 循环寻找最佳匹配点 ---
        while (true) {
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc); // 在当前的结果矩阵中查找最大值及其位置。

            if (maxVal >= similarity) { // 如果找到的最大相似度仍然高于或等于阈值。
                if (match_count > 0) { // 如果不是第一个匹配项，
                    coords_ss << "|"; // 则在前面加上分隔符 "|"。
                }
                coords_ss << (roiX + maxLoc.x) << "," << (roiY + maxLoc.y) << "," << maxVal; // 将找到的坐标（从ROI坐标转换为窗口坐标）和相似度拼接到字符串流中。
                match_count++; // 匹配计数器加一。
                
                // 关键步骤：在结果矩阵中将刚刚找到的区域"涂黑"（填充一个低值，这里是0），
                cv::rectangle(result, maxLoc, cv::Point(maxLoc.x + tpl.cols, maxLoc.y + tpl.rows), cv::Scalar(0), -1); // 这样在下一次循环中 minMaxLoc 就不会再次找到这个相同的位置。
            } else {
                break; // 如果结果矩阵中的最大值已经低于阈值，说明没有更多匹配项了，退出循环。
            }
        }

        // --- 4.4. 格式化最终结果字符串 "数量;坐标1|坐标2..." ---
        sprintf_s(resultBuffer, sizeof(resultBuffer), "%d;%s", match_count, coords_ss.str().c_str()); // 使用 sprintf_s 将匹配数量和坐标字符串格式化到最终的返回缓冲区中。

    } catch (...) {
        // 如果在 try 块中发生任何异常，函数将直接跳到末尾，返回默认的 "-1" 错误字符串。
    }

    // bmpBuffer 已被移除，无需释放
    return resultBuffer; // 返回最终的结果字符串。
}

/**
 * @brief 在指定窗口的特定区域(ROI)内，基于锚点分析网格布局，并返回每个单元格的状态。
 * 
 * 该函数首先在给定的ROI内截取图像，然后在该图像中寻找一个作为基准的"锚点"图片。
 * 以此锚点为起点，根据用户定义的网格行列、间距等参数，推算出每个单元格的位置，
 * 并进一步分析每个单元格内指定区域（如血条）的颜色状态，最终计算出百分比。
 * 
 * @param hwnd 目标窗口的句柄。
 * @param roiX, roiY, roiWidth, roiHeight 定义了在窗口客户区内进行分析的矩形区域 (Region of Interest)。
 * @param anchorTplPath 用于定位网格基准点的锚点模板图片路径。
 * @param similarity 查找锚点时所需的最低相似度 (0.0 - 1.0)。
 * @param gridRows, gridCols 网格的行数和列数。
 * @param firstCellOffsetX, firstCellOffsetY 第一个单元格左上角相对于锚点左上角的X、Y偏移量。
 * @param cellWidth, cellHeight 每个单元格的宽度和高度。
 * @param horizontalGap, verticalGap 单元格之间的水平和垂直间距。
 * @param healthBarOffsetX, healthBarOffsetY 每个单元格内部待分析区域（如血条）相对于单元格左上角的X、Y偏移量。
 * @param healthBarWidth, healthBarHeight 待分析区域（如血条）的宽度和高度。
 * 
 * @return const char* 一个字符串，格式为 "x1,y1,health1|x2,y2,health2|..."
 *         - x,y 是每个单元格中心点相对于【窗口】的坐标。
 *         - health 是从0到100的整数，代表分析区域的状态百分比。
 *         - 如果找不到锚点，返回 "-1"。
 *         - 如果发生其他错误（如参数无效、截图失败），返回 "-2"。
 *         - 如果未分析出任何单元格，返回空字符串 ""。
 */
static thread_local char g_gridResultBuffer[4096]; // 使用 thread_local 替代 static，确保线程安全

const char* __stdcall AnalyzeGrid(
    HWND hwnd,
    int roiX, int roiY, int roiWidth, int roiHeight,
    const char* anchorTplPath,
    double similarity,
    int gridRows,
    int gridCols,
    int firstCellOffsetX,
    int firstCellOffsetY,
    int cellWidth,
    int cellHeight,
    int horizontalGap,
    int verticalGap,
    int healthBarOffsetX,
    int healthBarOffsetY,
    int healthBarWidth,
    int healthBarHeight
) {
    // --- 0. 初始化和参数校验 ---
    const char* anchorNotFoundResult = "-1"; // 定义找不到锚点时的返回字符串。
    const char* errorResult = "-2"; // 定义通用错误（如参数错误、截图失败）的返回字符串。
    strcpy_s(g_gridResultBuffer, sizeof(g_gridResultBuffer), errorResult); // 默认将结果设置为通用错误码。

    // 对所有输入参数进行有效性检查。
    if (!IsWindow(hwnd) || !anchorTplPath || roiWidth <= 0 || roiHeight <= 0 || gridRows <= 0 || gridCols <= 0 || cellWidth <= 0 || cellHeight <= 0 || healthBarWidth <= 0 || healthBarHeight <= 0) {
        return g_gridResultBuffer; // 如果有任何参数无效，则返回错误。
    }

    // --- 1. 截图 (使用 Common 工具) ---
    cv::Mat screenMat;
    if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, screenMat)) {
        return g_gridResultBuffer;
    }
    // screenMat 已经是 BGR 格式，可以直接使用

    // --- 2. 定位锚点 (在 ROI 内) ---
    cv::Point anchorLoc; // 用于存储锚点在 ROI 内的左上角坐标。
    try {
        cv::Mat anchorTpl = cv::imread(anchorTplPath, cv::IMREAD_COLOR); // 加载锚点模板图片。
        if (anchorTpl.empty()) { // 如果模板加载失败。
            return g_gridResultBuffer;
        }

        cv::Mat matchResult; // 用于存储匹配结果。
        cv::matchTemplate(screenMat, anchorTpl, matchResult, cv::TM_CCOEFF_NORMED); // 在ROI截图中执行模板匹配。

        double minVal, maxVal; // 存储结果矩阵中的最小和最大相似度。
        cv::Point minLoc, maxLoc; // 存储最小和最大相似度的位置。
        cv::minMaxLoc(matchResult, &minVal, &maxVal, &minLoc, &maxLoc); // 查找最佳匹配位置。

        if (maxVal < similarity) { // 如果找到的最佳匹配相似度低于阈值。
            strcpy_s(g_gridResultBuffer, sizeof(g_gridResultBuffer), anchorNotFoundResult); // 设置结果为“未找到锚点”。
            return g_gridResultBuffer; // 返回。
        }
        anchorLoc = maxLoc; // 存储找到的锚点在ROI内的坐标。
    } catch (...) {
        return g_gridResultBuffer; // 如果发生OpenCV异常，返回通用错误。
    }

    // --- 3. 遍历网格并分析 ---
    std::stringstream resultStream; // 使用字符串流高效拼接结果。
    bool firstCell = true; // 标记是否是第一个被分析的单元格，用于控制分隔符'|'的添加。

    for (int row = 0; row < gridRows; ++row) { // 遍历每一行。
        for (int col = 0; col < gridCols; ++col) { // 遍历每一列。
            // --- 3.1. 计算当前单元格在 ROI 内的坐标 ---
            int cellX = anchorLoc.x + firstCellOffsetX + col * (cellWidth + horizontalGap); // 计算单元格左上角X坐标。
            int cellY = anchorLoc.y + firstCellOffsetY + row * (cellHeight + verticalGap); // 计算单元格左上角Y坐标。

            // --- 3.2. 计算血条区域在 ROI 内的坐标 ---
            int hbX = cellX + healthBarOffsetX; // 计算血条区域左上角X坐标。
            int hbY = cellY + healthBarOffsetY; // 计算血条区域左上角Y坐标。

            // --- 3.3. 边界检查，确保血条区域在 ROI 截图内 ---
            if (hbX < 0 || hbY < 0 || (hbX + healthBarWidth) > screenMat.cols || (hbY + healthBarHeight) > screenMat.rows) {
                continue; // 如果血条区域超出截图边界，则跳过当前单元格。
            }

            // --- 3.4. 提取血条区域并分析 ---
            cv::Rect healthBarRect(hbX, hbY, healthBarWidth, healthBarHeight); // 定义血条矩形区域。
            cv::Mat healthBarMat = screenMat(healthBarRect); // 从截图中提取血条区域。

            cv::Mat hsvMat; // 定义用于存储HSV图像的Mat。
            cv::cvtColor(healthBarMat, hsvMat, cv::COLOR_BGR2HSV); // 转换到HSV颜色空间，便于颜色分析。

            // --- 3.5. 通过腐蚀操作去除文字等噪声干扰 ---
            cv::Mat mask; // 定义二值蒙版。
            // 基于S(饱和度)和V(明度)通道创建二值蒙版，忽略H(色相)，以抵抗颜色变化。
            cv::inRange(hsvMat, cv::Scalar(0, 41, 41), cv::Scalar(180, 255, 255), mask);
            
            cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)); // 定义一个小的腐蚀核。
            cv::Mat erodedMask; // 定义腐蚀后的蒙版。
            cv::erode(mask, erodedMask, element); // 执行腐蚀操作，移除细小的噪声点（如文字）。
            
            int coloredPixels = cv::countNonZero(erodedMask); // 在干净的蒙版上计算非零像素数量，代表有效颜色区域。

            // --- 3.6. 计算百分比 ---
            int totalPixels = healthBarWidth * healthBarHeight; // 计算分析区域的总像素数。
            int health = (totalPixels > 0) ? (100 * coloredPixels / totalPixels) : 0; // 计算百分比。

            // --- 3.7. 计算格子中心点坐标 ---
            int centerX_roi = cellX + cellWidth / 2; // 计算单元格中心点在ROI内的X坐标。
            int centerY_roi = cellY + cellHeight / 2; // 计算单元格中心点在ROI内的Y坐标。

            // --- 3.8. 格式化输出 ---
            if (!firstCell) { // 如果不是第一个单元格。
                resultStream << "|"; // 添加分隔符。
            }
            // 将单元格中心点坐标（转换为窗口坐标）和计算出的百分比拼接到结果流中。
            resultStream << (roiX + centerX_roi) << "," << (roiY + centerY_roi) << "," << health;
            firstCell = false; // 更新标记。
        }
    }

    // --- 4. 返回最终结果 ---
    std::string finalResultStr = resultStream.str(); // 从字符串流获取最终结果。
    if (finalResultStr.empty()) { // 如果结果为空（没有分析任何单元格）。
        strcpy_s(g_gridResultBuffer, sizeof(g_gridResultBuffer), ""); // 设置为空字符串。
    } else {
        strcpy_s(g_gridResultBuffer, sizeof(g_gridResultBuffer), finalResultStr.c_str()); // 将结果复制到全局缓冲区。
    }

    return g_gridResultBuffer; // 返回结果。
}



const char* __stdcall Ocr(
    HWND hwnd,
    int roiX, int roiY, int roiWidth, int roiHeight,
    const char* lang,
    const char* tessdataPath)
{
    static thread_local std::string g_ocrResult;
    g_ocrResult = "-1";

    // --- 1. 参数校验 ---
    if (!IsWindow(hwnd) || roiWidth <= 0 || roiHeight <= 0 || !tessdataPath) {
        return g_ocrResult.c_str(); // 如果参数无效，直接返回错误。
    }

    // --- 2. 截图 (使用 Common 工具) ---
    cv::Mat bgraMat; // 注意：InternalOcr 可能需要 BGRA 或 BGR，这里 Common 返回 BGR
    // 但 InternalOcr 实现中处理了 3 通道和 4 通道。
    // Common::CaptureWindowRegion 返回 BGR (3通道)
    if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, bgraMat)) {
        return g_ocrResult.c_str();
    }

    // --- 4. 调用核心OCR函数 ---
    // 将截图得到的Mat图像传递给重构后的核心函数进行识别。
    std::vector<OcrWord> words = InternalOcr(bgraMat, lang, tessdataPath);

    // --- 5. 格式化输出 ---
    // 遍历识别结果，并将其格式化为 "文本1,x1,y1|文本2,x2,y2|..." 的字符串。
    std::stringstream out;
    for (const auto& word : words) {
        if (!word.text_utf8.empty()) {
            // 将UTF-8文本转换为本地ANSI编码以适应调用方
            std::string text_ansi = Common::Utf8ToAnsi(word.text_utf8);
            // 移除所有标点符号
            text_ansi.erase(std::remove_if(text_ansi.begin(), text_ansi.end(),
                [](unsigned char c) { return std::ispunct(c); }),
                text_ansi.end());
            
            if (!text_ansi.empty()) {
                // 计算单词中心点在目标窗口中的绝对坐标
                int centerX = roiX + word.x + word.width / 2;
                int centerY = roiY + word.y + word.height / 2;
                // 拼接字符串
                out << text_ansi << "," << centerX << "," << centerY << "|";
            }
        }
    }

    g_ocrResult = out.str();
    // 去掉字符串末尾可能多余的'|'
    if (!g_ocrResult.empty() && g_ocrResult.back() == '|') {
        g_ocrResult.pop_back();
    }

    return g_ocrResult.c_str(); // 返回最终格式化的字符串
}

/**
 * @brief 对指定的图像文件进行光学字符识别(OCR)，主要用于识别用户已预处理好的图片。
 * 
 * @param imgPath 待识别的图像文件路径。
 * @param lang Tesseract 使用的语言模型，例如 "eng", "chi_sim"。
 * @param tessdataPath 包含 .traineddata 语言文件的目录路径。
 * 
 * @return const char* - 识别出的文本字符串 (本地 ANSI 编码)。
 *                     - 如果未识别出任何文本，返回空字符串 ""。
 *                     - 如果发生错误（如文件加载失败），返回 "-1"。
 *                     - 如果 Tesseract 初始化失败，返回 "-2"。
 */
const char* __stdcall OcrFile(
    const char* imgPath,
    const char* lang,
    const char* tessdataPath)
{
    // 定义一个静态的 std::string 用于存储结果
    static thread_local std::string g_ocrFileResult;
    g_ocrFileResult = "-1"; // 默认设置为错误码

    // --- 1. 参数校验 ---
    if (!imgPath || !tessdataPath) {
        return g_ocrFileResult.c_str(); // 图像路径和Tesseract数据路径不能为空
    }

    // --- 2. 加载图像文件 ---
    // 使用OpenCV的imread函数从指定路径加载图像
    cv::Mat userImage = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (userImage.empty()) {
        return g_ocrFileResult.c_str(); // 如果图像加载失败或为空，返回错误
    }

    // --- 3. 调用核心OCR函数 ---
    // 将加载的图像直接传递给核心OCR函数进行处理
    std::vector<OcrWord> words = InternalOcr(userImage, lang, tessdataPath);

    // --- 4. 格式化输出 ---
    // 遍历识别结果，并将其格式化为 "文本1,x1,y1|文本2,x2,y2|..." 的字符串
    std::stringstream out;
    for (const auto& word : words) {
        if (!word.text_utf8.empty()) {
            // 将UTF-8文本转换为本地ANSI编码
            std::string text_ansi = Common::Utf8ToAnsi(word.text_utf8);
            // 移除所有标点符号
            text_ansi.erase(std::remove_if(text_ansi.begin(), text_ansi.end(),
                [](unsigned char c) { return std::ispunct(c); }),
                text_ansi.end());

            if (!text_ansi.empty()) {
                // 计算单词中心点在图像中的坐标
                int centerX = word.x + word.width / 2;
                int centerY = word.y + word.height / 2;
                // 拼接字符串
                out << text_ansi << "," << centerX << "," << centerY << "|";
            }
        }
    }

    g_ocrFileResult = out.str();
    // 去掉字符串末尾可能多余的'|'
    if (!g_ocrFileResult.empty() && g_ocrFileResult.back() == '|') {
        g_ocrFileResult.pop_back();
    }

    return g_ocrFileResult.c_str(); // 返回最终结果
}

/**
 * @brief 在指定坐标点进行颜色匹配，支持多颜色和模糊匹配。
 * 
 * @param hwnd 目标窗口的句柄。如果为 NULL 或无效句柄，则使用屏幕绝对坐标。
 *             如果提供有效句柄，则 x, y 被视为相对于该窗口客户区的坐标。
 * @param x, y 要拾取颜色的点的坐标。
 * @param colorStr 一个描述待匹配颜色的字符串，格式复杂：
 *                 "RRGGBB-Dev|RRGGBB-DevR,DevG,DevB|..."
 *                 - RRGGBB: 6位十六进制颜色值。
 *                 - Dev: 单一颜色偏差值，应用于R,G,B。
 *                 - DevR,DevG,DevB: 分别指定R,G,B的偏差。
 *                 - 多个颜色定义用 "|" 分隔。
 * 
 * @return const char* - 如果匹配成功，返回第一个匹配到的颜色在列表中的索引(从1开始)。
 *                       如果匹配到多个，则用 "|" 分隔，例如 "1|3"。
 *                     - 如果没有颜色匹配成功，返回 "0"。
 *                     - 如果发生错误（如参数无效、取色失败），返回 "-1"。
 */
const char* __stdcall FindColor(
    HWND hwnd,
    int x,
    int y,
    const char* colorStr
) {
    static thread_local char resultBuffer[1024];
    const char* errorResult = "-1"; // 定义表示错误的返回字符串
    const char* notFoundResult = "0"; // 定义表示未找到的返回字符串

    // --- 1. 参数校验 ---
    if (!colorStr || strlen(colorStr) == 0) { // 检查颜色字符串是否为空
        strcpy_s(resultBuffer, sizeof(resultBuffer), errorResult); // 如果为空，则直接返回错误
        return resultBuffer;
    }

    // --- 2. 获取目标坐标的颜色 ---
    COLORREF screenColor; // 用于存储从屏幕获取的颜色值 (COLORREF 是一个 DWORD)
    HDC hdc = NULL; // 设备上下文句柄

    if (hwnd != NULL && IsWindow(hwnd)) { // 检查是否提供了有效的窗口句柄
        // 相对坐标模式：在指定窗口的客户区内取色
        hdc = GetDC(hwnd); // 获取窗口的设备上下文
        if (!hdc) { // 获取失败
            strcpy_s(resultBuffer, sizeof(resultBuffer), errorResult);
            return resultBuffer;
        }
        screenColor = GetPixel(hdc, x, y); // 在窗口的 (x, y) 坐标取色
    } else {
        // 绝对坐标模式：在整个屏幕上取色
        hdc = GetDC(NULL); // 获取整个屏幕的设备上下文
        if (!hdc) { // 获取失败
            strcpy_s(resultBuffer, sizeof(resultBuffer), errorResult);
            return resultBuffer;
        }
        screenColor = GetPixel(hdc, x, y); // 在屏幕的 (x, y) 坐标取色
    }

    if (hdc) {
        ReleaseDC(hwnd, hdc); // 无论哪种模式，使用完毕后都要释放设备上下文
    }

    if (screenColor == CLR_INVALID) { // 检查 GetPixel 是否返回了无效颜色值
        strcpy_s(resultBuffer, sizeof(resultBuffer), errorResult);
        return resultBuffer;
    }

    // 从 COLORREF 中提取 R, G, B 分量
    int screenR = GetRValue(screenColor);
    int screenG = GetGValue(screenColor);
    int screenB = GetBValue(screenColor);

    // --- 3. 解析颜色字符串并进行匹配 ---
    std::string allColors(colorStr); // 将 C 风格字符串转换为 C++ string
    std::stringstream colorStream(allColors); // 使用 stringstream 以便按 '|' 分割
    std::string singleColorDef; // 用于存储单个颜色定义，例如 "FF0000-10"
    std::stringstream finalResultStream; // 用于拼接最终的结果字符串，例如 "1|3"
    bool firstMatch = true; // 标记是否是第一个匹配到的颜色
    int colorIndex = 1; // 颜色索引从1开始

    while (std::getline(colorStream, singleColorDef, '|')) { // 循环按 '|' 分割字符串
        size_t dashPos = singleColorDef.find('-'); // 查找颜色值和偏差值之间的分隔符 '-'
        if (dashPos == std::string::npos) { // 如果找不到分隔符
            colorIndex++;
            continue; // 格式错误，跳过当前颜色定义
        }

        std::string hexColorStr = singleColorDef.substr(0, dashPos); // 提取十六进制颜色字符串
        std::string deviationStr = singleColorDef.substr(dashPos + 1); // 提取偏差字符串

        if (hexColorStr.length() != 6 || deviationStr.empty()) { // 校验格式
            colorIndex++;
            continue; // 格式错误，跳过
        }

        try {
            // --- 3.1. 解析目标颜色和偏差 ---
            int targetColor = std::stoi(hexColorStr, nullptr, 16); // 将十六进制字符串转换为整数
            int devR, devG, devB; // 分别存储 R, G, B 的偏差

            // 智能解析偏差格式
            size_t commaPos1 = deviationStr.find(','); // 查找偏差字符串中的逗号
            if (commaPos1 != std::string::npos) {
                // 格式: "DR,DG,DB"
                size_t commaPos2 = deviationStr.find(',', commaPos1 + 1); // 查找第二个逗号
                if (commaPos2 != std::string::npos) {
                    devR = std::stoi(deviationStr.substr(0, commaPos1));
                    devG = std::stoi(deviationStr.substr(commaPos1 + 1, commaPos2 - (commaPos1 + 1)));
                    devB = std::stoi(deviationStr.substr(commaPos2 + 1));
                } else {
                    colorIndex++; continue; // 格式错误，例如 "10,20"，缺少第三个值
                }
            } else {
                // 格式: "D" (单一偏差)
                devR = devG = devB = std::stoi(deviationStr); // 将单一偏差值赋给 R, G, B
            }

            // 从整数颜色值中提取 R, G, B 分量
            int targetR = (targetColor >> 16) & 0xFF;
            int targetG = (targetColor >> 8) & 0xFF;
            int targetB = targetColor & 0xFF;

            // --- 3.2. 执行模糊颜色比较 ---
            // 检查屏幕上实际颜色的 R, G, B 值是否都在目标颜色加减偏差的范围内
            if (abs(screenR - targetR) <= devR &&
                abs(screenG - targetG) <= devG &&
                abs(screenB - targetB) <= devB)
            {
                if (!firstMatch) { // 如果不是第一个匹配项
                    finalResultStream << "|"; // 在前面加上分隔符
                }
                finalResultStream << colorIndex; // 将当前颜色索引追加到结果流
                firstMatch = false; // 更新标记
            }
        } catch (...) {
            // 如果 std::stoi 转换失败（例如，字符串包含非数字字符），捕获异常并忽略此颜色定义
        }
        colorIndex++; // 递增颜色索引，为下一个颜色定义做准备
    }

    // --- 4. 格式化并返回最终结果 ---
    std::string finalResultStr = finalResultStream.str(); // 从结果流中获取字符串
    if (finalResultStr.empty()) { // 如果字符串为空，说明没有匹配到任何颜色
        strcpy_s(resultBuffer, sizeof(resultBuffer), notFoundResult); // 返回 "0"
    } else {
        strcpy_s(resultBuffer, sizeof(resultBuffer), finalResultStr.c_str()); // 返回拼接好的匹配索引字符串
    }

    return resultBuffer;
}

// --- 内存读取功能 ---

// NtWow64ReadVirtualMemory64 函数类型定义，用于32位进程读取64位进程内存
typedef NTSTATUS (NTAPI *pfnNtWow64ReadVirtualMemory64)(
    HANDLE ProcessHandle,
    ULONGLONG BaseAddress,
    PVOID Buffer,
    ULONGLONG Size,
    PULONGLONG NumberOfBytesRead
);

// NtWow64QueryInformationProcess64 函数类型定义，用于查询64位进程信息
typedef NTSTATUS (NTAPI *pfnNtWow64QueryInformationProcess64)(
    HANDLE ProcessHandle,
    ULONG ProcessInformationClass,
    PVOID ProcessInformation,
    ULONG ProcessInformationLength,
    PULONG ReturnLength
);

// 64位进程的PEB结构（简化版，只包含需要的字段）
typedef struct _PEB64 {
    BYTE Reserved1[2];
    BYTE BeingDebugged;
    BYTE Reserved2[21];
    ULONGLONG Ldr;  // 偏移 0x18: PEB_LDR_DATA64 指针
    // ... 其他字段省略
} PEB64;

// 64位进程的 PROCESS_BASIC_INFORMATION
typedef struct _PROCESS_BASIC_INFORMATION64 {
    ULONGLONG Reserved1;
    ULONGLONG PebBaseAddress;
    ULONGLONG Reserved2[2];
    ULONGLONG UniqueProcessId;
    ULONGLONG Reserved3;
} PROCESS_BASIC_INFORMATION64;

// 64位 PEB_LDR_DATA 结构
typedef struct _PEB_LDR_DATA64 {
    ULONG Length;
    ULONG Initialized;
    ULONGLONG SsHandle;
    LIST_ENTRY64 InLoadOrderModuleList;
    LIST_ENTRY64 InMemoryOrderModuleList;
    LIST_ENTRY64 InInitializationOrderModuleList;
} PEB_LDR_DATA64;

// 64位 LDR_DATA_TABLE_ENTRY 结构
typedef struct _LDR_DATA_TABLE_ENTRY64 {
    LIST_ENTRY64 InLoadOrderLinks;
    LIST_ENTRY64 InMemoryOrderLinks;
    LIST_ENTRY64 InInitializationOrderLinks;
    ULONGLONG DllBase;
    ULONGLONG EntryPoint;
    ULONG SizeOfImage;
    ULONG Reserved;
    UNICODE_STRING64 FullDllName;
    UNICODE_STRING64 BaseDllName;
} LDR_DATA_TABLE_ENTRY64;

// 定义 UNICODE_STRING64（如果未定义）
#ifndef _UNICODE_STRING64_DEFINED
#define _UNICODE_STRING64_DEFINED
typedef struct _UNICODE_STRING64 {
    USHORT Length;
    USHORT MaximumLength;
    ULONGLONG Buffer;
} UNICODE_STRING64;
#endif

// 定义 LIST_ENTRY64（如果未定义）
#ifndef _LIST_ENTRY64_DEFINED
#define _LIST_ENTRY64_DEFINED
typedef struct _LIST_ENTRY64 {
    ULONGLONG Flink;
    ULONGLONG Blink;
} LIST_ENTRY64;
#endif

// 全局函数指针，延迟加载
static pfnNtWow64ReadVirtualMemory64 g_NtWow64ReadVirtualMemory64 = nullptr;
static pfnNtWow64QueryInformationProcess64 g_NtWow64QueryInformationProcess64 = nullptr;
static bool g_Wow64ApiInitialized = false;

// 初始化 Wow64 API
static bool InitWow64Api() {
    if (g_Wow64ApiInitialized) {
        return g_NtWow64ReadVirtualMemory64 != nullptr;
    }
    g_Wow64ApiInitialized = true;
    
    HMODULE hNtdll = GetModuleHandleA("ntdll.dll");
    if (hNtdll) {
        g_NtWow64ReadVirtualMemory64 = (pfnNtWow64ReadVirtualMemory64)
            GetProcAddress(hNtdll, "NtWow64ReadVirtualMemory64");
        g_NtWow64QueryInformationProcess64 = (pfnNtWow64QueryInformationProcess64)
            GetProcAddress(hNtdll, "NtWow64QueryInformationProcess64");
    }
    return g_NtWow64ReadVirtualMemory64 != nullptr;
}

// 跨位数读取内存的封装函数
static bool CrossReadProcessMemory(
    HANDLE hProcess,
    ULONGLONG address,
    PVOID buffer,
    SIZE_T size,
    bool is64BitTarget
) {
    SIZE_T bytesRead = 0;
    
    if (is64BitTarget) {
        // 64位目标进程，使用 NtWow64ReadVirtualMemory64
        if (!g_NtWow64ReadVirtualMemory64) {
            return false;
        }
        ULONGLONG bytesRead64 = 0;
        NTSTATUS status = g_NtWow64ReadVirtualMemory64(
            hProcess, address, buffer, size, &bytesRead64);
        return (status >= 0) && (bytesRead64 == size);
    } else {
        // 32位目标进程，使用标准 ReadProcessMemory
        return ReadProcessMemory(hProcess, (LPCVOID)(DWORD)address, buffer, size, &bytesRead) 
               && (bytesRead == size);
    }
}

/**
 * @brief 获取64位进程中特定模块的基地址（通过读取PEB64）
 */
static ULONGLONG GetModuleBaseAddress64Bit(HANDLE hProcess, const char* moduleName) {
    if (!g_NtWow64QueryInformationProcess64 || !g_NtWow64ReadVirtualMemory64) {
        return 0;
    }
    
    // 1. 获取64位进程的PEB地址
    PROCESS_BASIC_INFORMATION64 pbi64 = {0};
    ULONG returnLength = 0;
    NTSTATUS status = g_NtWow64QueryInformationProcess64(
        hProcess, 
        0, // ProcessBasicInformation
        &pbi64, 
        sizeof(pbi64), 
        &returnLength
    );
    
    if (status < 0 || pbi64.PebBaseAddress == 0) {
        return 0;
    }
    
    // 2. 读取PEB64获取Ldr指针
    ULONGLONG ldrAddress = 0;
    ULONGLONG bytesRead = 0;
    
    // PEB64.Ldr 在偏移 0x18 处
    status = g_NtWow64ReadVirtualMemory64(
        hProcess,
        pbi64.PebBaseAddress + 0x18,
        &ldrAddress,
        sizeof(ldrAddress),
        &bytesRead
    );
    
    if (status < 0 || ldrAddress == 0) {
        return 0;
    }
    
    // 3. 读取 PEB_LDR_DATA64 获取模块链表头
    PEB_LDR_DATA64 ldrData = {0};
    status = g_NtWow64ReadVirtualMemory64(
        hProcess,
        ldrAddress,
        &ldrData,
        sizeof(ldrData),
        &bytesRead
    );
    
    if (status < 0) {
        return 0;
    }
    
    // 4. 遍历 InLoadOrderModuleList 链表
    ULONGLONG listHead = ldrAddress + offsetof(PEB_LDR_DATA64, InLoadOrderModuleList);
    ULONGLONG currentEntry = ldrData.InLoadOrderModuleList.Flink;
    
    // 转换模块名为宽字符以便比较
    wchar_t wModuleName[256] = {0};
    MultiByteToWideChar(CP_ACP, 0, moduleName, -1, wModuleName, 256);
    _wcslwr_s(wModuleName, 256); // 转小写
    
    int maxIterations = 1000; // 防止无限循环
    while (currentEntry != listHead && maxIterations-- > 0) {
        // 读取 LDR_DATA_TABLE_ENTRY64
        LDR_DATA_TABLE_ENTRY64 entry = {0};
        status = g_NtWow64ReadVirtualMemory64(
            hProcess,
            currentEntry,
            &entry,
            sizeof(entry),
            &bytesRead
        );
        
        if (status < 0) {
            break;
        }
        
        // 读取模块名
        if (entry.BaseDllName.Length > 0 && entry.BaseDllName.Buffer != 0) {
            wchar_t dllName[256] = {0};
            USHORT nameLen = min(entry.BaseDllName.Length, (USHORT)(sizeof(dllName) - 2));
            
            status = g_NtWow64ReadVirtualMemory64(
                hProcess,
                entry.BaseDllName.Buffer,
                dllName,
                nameLen,
                &bytesRead
            );
            
            if (status >= 0) {
                dllName[nameLen / sizeof(wchar_t)] = L'\0';
                _wcslwr_s(dllName, 256); // 转小写
                
                // 比较模块名
                if (wcscmp(dllName, wModuleName) == 0) {
                    return entry.DllBase;
                }
            }
        }
        
        // 移动到下一个条目
        currentEntry = entry.InLoadOrderLinks.Flink;
    }
    
    return 0;
}

/**
 * @brief 获取指定进程中特定模块的基地址（支持32位和64位进程）
 * 
 * @param hProcess 已打开的进程句柄
 * @param processId 目标进程的ID
 * @param moduleName 要查找的模块名称
 * @param is64BitProcess 目标进程是否为64位
 * @return ULONGLONG 模块基地址，失败返回0
 */
static ULONGLONG GetModuleBaseAddressEx(HANDLE hProcess, DWORD processId, const char* moduleName, bool is64BitProcess) {
    if (is64BitProcess) {
        // 64位进程：通过PEB64获取模块基址
        return GetModuleBaseAddress64Bit(hProcess, moduleName);
    } else {
        // 32位进程：使用标准的 CreateToolhelp32Snapshot
        HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, processId);
        if (hSnap == INVALID_HANDLE_VALUE) {
            return 0;
        }
        
        MODULEENTRY32 me32;
        me32.dwSize = sizeof(MODULEENTRY32);
        
        ULONGLONG result = 0;
        if (Module32First(hSnap, &me32)) {
            do {
                if (_stricmp(me32.szModule, moduleName) == 0) {
                    result = (ULONGLONG)(uintptr_t)me32.modBaseAddr;
                    break;
                }
            } while (Module32Next(hSnap, &me32));
        }
        
        CloseHandle(hSnap);
        return result;
    }
}

/**
 * @brief 获取指定进程中特定模块的基地址。
 * 
 * @param processId 目标进程的ID。
 * @param moduleName 要查找的模块名称 (例如 "kernel32.dll" 或 "MyGame.exe")。
 * @return uintptr_t 如果找到，返回模块的基地址；否则返回 0。
 */
uintptr_t GetModuleBaseAddress(DWORD processId, const char* moduleName) {
    // --- 1. 创建进程的模块快照 ---
    // TH32CS_SNAPMODULE: 包含所有模块
    // TH32CS_SNAPMODULE32: 在64位进程中获取32位模块信息
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, processId);
    if (hSnap == INVALID_HANDLE_VALUE) { // 检查快照句柄是否有效
        return 0; // 创建失败，返回0
    }

    // --- 2. 初始化模块入口结构体 ---
    MODULEENTRY32 me32;
    me32.dwSize = sizeof(MODULEENTRY32); // 在调用 Module32First 之前必须设置此成员

    // --- 3. 遍历模块列表 ---
    if (Module32First(hSnap, &me32)) { // 获取第一个模块的信息
        do {
            // 使用 _stricmp 进行不区分大小写的比较，以增加匹配的灵活性
            if (_stricmp(me32.szModule, moduleName) == 0) {
                // --- 4. 找到匹配项，清理并返回 ---
                CloseHandle(hSnap); // 关闭快照句柄，释放资源
                return (uintptr_t)me32.modBaseAddr; // 返回找到的模块基地址
            }
        } while (Module32Next(hSnap, &me32)); // 继续遍历下一个模块
    }

    // --- 5. 未找到，清理并返回 ---
    CloseHandle(hSnap); // 如果循环结束仍未找到，关闭快照句柄
    return 0; // 返回0表示未找到
}

/**
 * @brief 根据给定的地址表达式，从目标进程读取内存数据。
 * 
 * @param hwnd 目标窗口的句柄，用于获取进程ID。
 * @param addressExpression 内存地址表达式，格式为 "模块名+偏移1[+偏移2[+...]]"。
 *                          例如: "MyGame.exe+AABBCC" 或 "MyGame.exe+AABBCC+10+20"。
 * @param readType 要读取的数据类型，支持:
 *                 - "i": 整型 (根据 readSize 决定是 1, 2, 4, 8 字节)
 *                 - "f": 浮点型 (4 字节)
 *                 - "d": 双精度浮点型 (8 字节)
 *                 - "s": 字符串 (ANSI)
 *                 - "w": 宽字符串 (UTF-16)
 * @param readSize 要读取的字节数。对于字符串类型，这指定了最大长度。
 * 
 * @return const char* - 成功读取并格式化后的字符串。
 *                     - 如果失败，返回 "-1"。
 */
const char* __stdcall ReadMemory(
    HWND hwnd,
    const char* addressExpression,
    const char* readType,
    int readSize
) {
    // 定义一个静态缓冲区来存储返回的字符串，确保函数返回后指针依然有效。
    static thread_local char resultBuffer[1024];
    // 每次调用前，默认将结果设置为错误码 "-1"。
    strcpy_s(resultBuffer, sizeof(resultBuffer), "-1");

    // --- 1. 参数校验 ---
    // 检查窗口句柄是否有效，地址表达式和读取类型指针是否为空，以及读取大小是否为正数。
    if (!IsWindow(hwnd) || !addressExpression || !readType || readSize <= 0) {
        return resultBuffer; // 如果参数无效，直接返回默认的错误码。
    }

    // --- 2. 获取进程句柄 ---
    DWORD processId = 0; // 用于存储目标进程的ID。
    GetWindowThreadProcessId(hwnd, &processId); // 从窗口句柄获取其所属进程的ID。
    if (processId == 0) { // 如果获取失败。
        return resultBuffer; // 返回错误。
    }

    // 使用 OpenProcess 打开目标进程，请求读取内存(PROCESS_VM_READ)和查询信息(PROCESS_QUERY_INFORMATION)的权限。
    HANDLE hProcess = OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, FALSE, processId);
    if (hProcess == NULL) { // 如果打开进程失败。
        return resultBuffer; // 返回错误。
    }

    // --- 3. 检测目标进程位数并初始化Wow64 API ---
    BOOL isWow64Target = FALSE;
    IsWow64Process(hProcess, &isWow64Target);
    bool is64BitTarget = !isWow64Target; // isWow64=FALSE 表示是原生64位进程
    
    // 如果目标是64位进程，初始化Wow64 API
    if (is64BitTarget && !InitWow64Api()) {
        CloseHandle(hProcess);
        return resultBuffer; // 无法初始化64位读取API
    }

    // --- 4. 解析地址表达式 ---
    char expr_copy[512]; // 创建一个可修改的副本，因为 strtok_s 会修改原字符串。
    strcpy_s(expr_copy, sizeof(expr_copy), addressExpression);

    char* context = NULL; // strtok_s 需要的上下文指针。
    // 使用 "[]+" 作为分隔符来解析表达式，例如 "Game.exe+123[456]" 会被分割成 "Game.exe", "123", "456"。
    char* token = strtok_s(expr_copy, "[]+", &context);
    if (token == NULL) { CloseHandle(hProcess); return resultBuffer; } // 第一个 token 应该是模块名，如果为空则表达式无效。

    // --- 4.1. 获取模块基地址 ---
    ULONGLONG finalAddress = 0; // 使用64位地址存储，兼容32/64位进程
    ULONGLONG moduleBase = GetModuleBaseAddressEx(hProcess, processId, token, is64BitTarget);
    if (moduleBase == 0) { CloseHandle(hProcess); return resultBuffer; } // 如果找不到模块，则无法继续。

    // --- 4.2. 计算初始地址 (基地址 + 第一个偏移) ---
    token = strtok_s(NULL, "[]+", &context); // 获取下一个 token，即第一个偏移量。
    if (token == NULL) { CloseHandle(hProcess); return resultBuffer; } // 表达式必须至少有一个偏移量。
    finalAddress = moduleBase + std::stoull(token, nullptr, 16); // 将16进制的偏移字符串转换为数字，并与基地址相加。

    // --- 4.3. 循环处理多级指针偏移 ---
    // 循环获取后续的所有偏移量，直到没有更多 token。
    while ((token = strtok_s(NULL, "[]+", &context)) != NULL) {
        ULONGLONG temp_ptr = 0; // 临时变量，用于存储从内存中读取到的指针地址。
        
        if (is64BitTarget) {
            // 64位目标进程，读取8字节指针
            ULONGLONG ptr64 = 0;
            if (!CrossReadProcessMemory(hProcess, finalAddress, &ptr64, sizeof(ptr64), true)) {
                CloseHandle(hProcess); return resultBuffer;
            }
            temp_ptr = ptr64;
        } else {
            // 32位目标进程，读取4字节指针
            DWORD ptr32 = 0;
            if (!CrossReadProcessMemory(hProcess, finalAddress, &ptr32, sizeof(ptr32), false)) {
                CloseHandle(hProcess); return resultBuffer;
            }
            temp_ptr = ptr32;
        }
        
        // 计算下一级地址：finalAddress = (上一步读到的指针值) + (当前偏移量)。
        finalAddress = temp_ptr + std::stoull(token, nullptr, 16);
    }

    // --- 5. 读取最终地址的数据并格式化 ---
    char readBuffer[512] = { 0 }; // 定义一个缓冲区来存储从内存中读取的原始数据。
    // 从计算出的 finalAddress 读取指定大小的数据。
    if (!CrossReadProcessMemory(hProcess, finalAddress, readBuffer, readSize, is64BitTarget)) {
        CloseHandle(hProcess); // 如果读取失败。
        return resultBuffer; // 返回错误。
    }

    std::string type(readType); // 将读取类型转换为 std::string 以便比较。
    if (type == "i") { // 如果是整型。
        if (readSize == 1) sprintf_s(resultBuffer, sizeof(resultBuffer), "%d", *(int8_t*)readBuffer);
        else if (readSize == 2) sprintf_s(resultBuffer, sizeof(resultBuffer), "%d", *(int16_t*)readBuffer);
        else if (readSize == 4) sprintf_s(resultBuffer, sizeof(resultBuffer), "%d", *(int32_t*)readBuffer);
        else if (readSize == 8) sprintf_s(resultBuffer, sizeof(resultBuffer), "%lld", *(int64_t*)readBuffer);
        else { CloseHandle(hProcess); return resultBuffer; } // 不支持的整型大小。
    } else if (type == "f") { // 如果是单精度浮点型。
        sprintf_s(resultBuffer, sizeof(resultBuffer), "%f", *(float*)readBuffer);
    } else if (type == "d") { // 如果是双精度浮点型。
        sprintf_s(resultBuffer, sizeof(resultBuffer), "%lf", *(double*)readBuffer);
    } else if (type == "s") { // 如果是ANSI字符串。
        readBuffer[readSize] = '\0'; // 确保字符串以 null 结尾，防止缓冲区溢出。
        strcpy_s(resultBuffer, sizeof(resultBuffer), readBuffer);
    } else if (type == "w") { // 如果是宽字符串 (UTF-16)。
        wchar_t wideBuffer[256] = { 0 }; // 定义一个宽字符缓冲区。
        // 将读取的原始数据复制到宽字符缓冲区，并确保不会溢出。
        memcpy(wideBuffer, readBuffer, std::min((size_t)readSize, sizeof(wideBuffer) - sizeof(wchar_t)));
        // 将读取到的宽字符串转换为调用方更易处理的编码（这里是UTF-8）。
        int n = WideCharToMultiByte(CP_UTF8, 0, wideBuffer, -1, resultBuffer, (int)sizeof(resultBuffer), nullptr, nullptr);
        if (n <= 0) { CloseHandle(hProcess); return resultBuffer; } // 转换失败。
    } else {
        CloseHandle(hProcess); // 未知的读取类型。
        return resultBuffer;
    }

    // --- 6. 清理并返回 ---
    CloseHandle(hProcess); // 关闭进程句柄，释放资源。
    return resultBuffer; // 返回格式化后的结果字符串。
}

/**
 * @brief 在给定的内存图像数据中查找一个模板图片。
 * 
 * @param imgData 指向图像原始像素数据的指针 (例如，来自截图的 BGRA 缓冲区)。
 * @param imgWidth 图像的宽度。
 * @param imgHeight 图像的高度。
 * @param imgChannels 图像的通道数 (通常是 3 for BGR, 4 for BGRA)。
 * @param tplPath 模板图片的文件路径。
 * @param similarity 相似度阈值 (0.0 - 1.0)。
 * @param x [out] 如果找到，返回匹配位置左上角相对于内存图像的X坐标。
 * @param y [out] 如果找到，返回匹配位置左上角相对于内存图像的Y坐标。
 * @param confidence [out] 如果找到，返回实际的匹配相似度。
 * 
 * @return int - 1: 成功找到。
 *             - 0: 未找到或参数无效。
 *             - -1: 发生未知异常。
 */
int __stdcall FindImageFromMem(
    const unsigned char* imgData,
    int imgWidth,
    int imgHeight,
    int imgChannels,
    const char* tplPath,
    double similarity,
    int* x,
    int* y,
    double* confidence)
{
    // --- 1. 参数校验和初始化 ---
    if (!imgData || !tplPath || !x || !y || !confidence) return 0; // 检查所有指针是否有效
    if (imgWidth <= 0 || imgHeight <= 0 || (imgChannels != 3 && imgChannels != 4)) return 0; // 检查图像尺寸和通道数是否合法
    *x = *y = -1; // 初始化输出坐标为-1
    *confidence = 0.0; // 初始化置信度为0.0

    try {
        // --- 2. 准备主图像 ---
        // 将内存中的像素数据包装成一个 cv::Mat 对象。这是一个“零拷贝”操作，非常高效。
        // 它不会复制数据，而是让 Mat 头指向现有的内存缓冲区。
        cv::Mat img(imgHeight, imgWidth, CV_8UC(imgChannels), (void*)imgData);

        // --- 3. 加载模板图像 ---
        cv::Mat tpl = cv::imread(tplPath, cv::IMREAD_COLOR); // 从文件加载模板

        if (img.empty() || tpl.empty()) return 0; // 检查主图或模板是否加载成功

        // --- 4. 确保通道数一致 ---
        // 模板匹配要求主图和模板图有相同的通道数。
        cv::Mat imgToMatch = img;
        if (img.channels() == 4) { // 如果主图是4通道 (例如 BGRA)
            cv::cvtColor(img, imgToMatch, cv::COLOR_BGRA2BGR); // 将其转换为3通道 (BGR)
        }
        
        cv::Mat tplToMatch = tpl;
        if (tpl.channels() == 4) { // 如果模板图也是4通道
            cv::cvtColor(tpl, tplToMatch, cv::COLOR_BGRA2BGR); // 也转换为3通道
        }

        // --- 5. 执行模板匹配 ---
        cv::Mat result;
        // 使用归一化相关系数方法进行匹配
        cv::matchTemplate(imgToMatch, tplToMatch, result, cv::TM_CCOEFF_NORMED);

        // --- 6. 查找最佳匹配点 ---
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc); // 查找结果矩阵中的最大值及其位置

        // --- 7. 检查相似度并返回结果 ---
        if (maxVal >= similarity) { // 如果最大相似度满足阈值
            *x = maxLoc.x; // 设置输出X坐标
            *y = maxLoc.y; // 设置输出Y坐标
            *confidence = maxVal; // 设置输出置信度
            return 1; // 返回成功
        }
    } catch (...) {
        return -1; // 捕获任何OpenCV异常并返回错误码
    }
    return 0; // 如果 try 块正常结束但未找到匹配，返回0
}

/**
 * @brief 在指定窗口的ROI内，一次性查找多个模板图片的所有匹配项。
 * 
 * @param hwnd 目标窗口的句柄。
 * @param roiX, roiY, roiWidth, roiHeight 定义在窗口客户区内进行查找的矩形区域。
 * @param multiTplPaths 包含多个模板图片路径的字符串，路径之间用 "|" 分隔。
 *                      例如: "C:\\img\\a.png|C:\\img\\b.png"
 * @param similarity 相似度阈值 (0.0 - 1.0)。
 * 
 * @return const char* 一个字符串，格式为 "文件名1,x1,y1,sim1|文件名2,x2,y2,sim2|..."
 *         - 如果未找到任何匹配项，返回空字符串 ""。
 *         - 如果发生错误，返回 "-1"。
 */
const char* __stdcall CaptureAndFindMultiTemplates(
    HWND hwnd,
    int roiX, int roiY, int roiWidth, int roiHeight,
    const char* multiTplPaths,
    double similarity
) {
    // 使用 thread_local 确保线程安全
    static thread_local char resultBuffer[4096];
    const char* errorResult = "-1";
    const char* notFoundResult = "";

    // --- 1. 参数校验 ---
    if (!IsWindow(hwnd) || !multiTplPaths || strlen(multiTplPaths) == 0 || roiWidth <= 0 || roiHeight <= 0) {
        strcpy_s(resultBuffer, sizeof(resultBuffer), errorResult);
        return resultBuffer;
    }

    // --- 2. 截图 (使用 Common 工具，避免代码重复和内存泄漏) ---
    cv::Mat screenMat;
    if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, screenMat)) {
        strcpy_s(resultBuffer, sizeof(resultBuffer), errorResult);
        return resultBuffer;
    }

    // --- 3. 解析模板路径并循环查找 ---
    std::string allPaths(multiTplPaths);
    std::stringstream pathStream(allPaths);
    std::string singlePath;
    std::stringstream finalResultStream;
    bool firstMatch = true;

    while (std::getline(pathStream, singlePath, '|')) {
        if (singlePath.empty()) continue;

        try {
            cv::Mat tpl = cv::imread(singlePath, cv::IMREAD_COLOR);
            if (tpl.empty()) continue;

            // 从完整路径中提取文件名
            std::string filename = singlePath.substr(singlePath.find_last_of("/\\") + 1);

            cv::Mat matchResult;
            cv::matchTemplate(screenMat, tpl, matchResult, cv::TM_CCOEFF_NORMED);

            double minVal, maxVal;
            cv::Point minLoc, maxLoc;

            // 循环查找当前模板的所有匹配项
            while (true) {
                cv::minMaxLoc(matchResult, &minVal, &maxVal, &minLoc, &maxLoc);
                if (maxVal >= similarity) {
                    if (!firstMatch) {
                        finalResultStream << "|";
                    }
                    finalResultStream << filename << "," << (roiX + maxLoc.x) << "," << (roiY + maxLoc.y) << "," << maxVal;
                    firstMatch = false;
                    
                    // 关键步骤：在结果矩阵中将刚刚找到的区域“涂黑”（填充一个低值，这里是0），
                    // 这样在下一次循环中 minMaxLoc 就不会再次找到这个相同或重叠的位置。
                    cv::rectangle(matchResult, maxLoc, cv::Point(maxLoc.x + tpl.cols, maxLoc.y + tpl.rows), cv::Scalar(0), -1);
                } else {
                    break; // 如果当前结果矩阵中的最大值已经低于阈值，说明这个模板没有更多匹配项了，退出内层循环。
                }
            }
        } catch (...) {
            // 忽略单个模板在加载或匹配过程中可能发生的OpenCV异常，继续处理下一个模板。
            continue;
        }
    }

    // --- 6. 格式化并返回最终结果 ---
    std::string finalResultStr = finalResultStream.str(); // 从字符串流获取最终结果。
    if (finalResultStr.empty()) { // 如果结果为空。
        strcpy_s(resultBuffer, sizeof(resultBuffer), notFoundResult); // 设置为未找到的结果。
    } else {
        strcpy_s(resultBuffer, sizeof(resultBuffer), finalResultStr.c_str()); // 将结果复制到静态缓冲区。
    }

    return resultBuffer; // 返回最终结果。
}

/**
 * @brief 在指定窗口的特定区域(ROI)内进行OCR，并查找特定文本字符串的位置。
 *
 * @param hwnd 目标窗口的句柄。
 * @param roiX, roiY, roiWidth, roiHeight 定义在窗口客户区内进行查找的矩形区域。
 * @param textToFind 要查找的目标文本 (UTF-8 编码)。
 * @param lang Tesseract 使用的语言模型，例如 "chi_sim"。
 * @param tessdataPath 包含 .traineddata 语言文件的目录路径。
 *
 * @return const char* - 成功时返回 "x,y" 格式的中心坐标字符串。
 *                     - 未找到文本时返回 "0"。
 *                     - 发生错误时返回 "-1"。
 */
HHAPI const char* __stdcall HH_FindText(
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    const char* textToFind,
    const char* lang,
    const char* tessdataPath
) {
    // 定义一个静态缓冲区来存储返回的坐标字符串
    static thread_local char g_findTextResultBuffer[256];
    strcpy_s(g_findTextResultBuffer, sizeof(g_findTextResultBuffer), "-1"); // 默认设置为错误码

    // --- 1. 参数校验 ---
    if (!IsWindow(hwnd) || roiWidth <= 0 || roiHeight <= 0 || !textToFind || !tessdataPath) {
        return g_findTextResultBuffer;
    }

    // --- 2. 截图 (使用 Common 工具) ---
    cv::Mat bgraMat;
    if (!Common::CaptureWindowRegion(hwnd, roiX, roiY, roiWidth, roiHeight, bgraMat)) {
        return g_findTextResultBuffer;
    }

    // --- 4. 调用核心OCR函数 ---
    // 将截图得到的Mat图像传递给核心函数进行识别，获取所有单词
    std::vector<OcrWord> words = InternalOcr(bgraMat, lang, tessdataPath);

    // --- 5. 查找目标文本 ---
    bool found = false;
    int found_x = -1, found_y = -1;

    // 将调用者传入的ANSI编码的目标文本转换为宽字符串(wstring)，以便进行可靠的包含检查
    std::wstring textToFindW = Common::AnsiToWide(textToFind);
    if (textToFindW.empty()) {
        strcpy_s(g_findTextResultBuffer, sizeof(g_findTextResultBuffer), "0"); // 如果要找的文本为空，直接返回未找到
        return g_findTextResultBuffer;
    }

    // 遍历从核心函数返回的所有识别出的单词
    for (const auto& word : words) {
        // 将识别出的UTF-8单词也转换为宽字符串
        std::wstring recognizedWordW = Common::Utf8ToWide(word.text_utf8);
        // 在宽字符串环境下，使用find方法检查识别出的单词是否包含目标文本
        if (recognizedWordW.find(textToFindW) != std::wstring::npos) {
            // 如果找到，计算其中心点在目标窗口中的绝对坐标
            found_x = roiX + word.x + word.width / 2;
            found_y = roiY + word.y + word.height / 2;
            found = true; // 设置找到标志
            break; // 找到第一个匹配项后立即退出循环
        }
    }

    // --- 6. 格式化并返回结果 ---
    if (found) {
        // 如果找到，将坐标格式化为 "x,y" 字符串
        sprintf_s(g_findTextResultBuffer, sizeof(g_findTextResultBuffer), "%d,%d", found_x, found_y);
    } else {
        // 如果未找到，返回 "0"
        strcpy_s(g_findTextResultBuffer, sizeof(g_findTextResultBuffer), "0");
    }

    return g_findTextResultBuffer; // 返回最终结果
}



