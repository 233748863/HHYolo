#pragma once
#include <string>
#include <windows.h>
#include <opencv2/opencv.hpp>

namespace Common {

    // --- OCR 预处理常量 ---
    constexpr float OCR_SCALE_FACTOR = 3.0f;           // OCR图像放大倍数
    constexpr int OCR_BORDER_SIZE = 10;                // OCR图像边框大小
    constexpr double OCR_LOW_CONTRAST_THRESHOLD = 30.0; // 低对比度阈值

    // --- String Conversions ---
    std::wstring Utf8ToWide(const std::string& s);
    std::string WideToAnsi(const std::wstring& w);
    std::string Utf8ToAnsi(const std::string& s);
    std::wstring AnsiToWide(const std::string& s);

    // --- File Operations ---
    bool FileExists(const char* filePath);

    // --- Window Operations ---
    bool CaptureWindowRegion(
        HWND hwnd,
        int roiX,
        int roiY,
        int roiWidth,
        int roiHeight,
        cv::Mat& outputBgr
    );

} // namespace Common
