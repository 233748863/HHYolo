#pragma once
#include <string>
#include <windows.h>
#include <opencv2/opencv.hpp>

namespace Common {

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
