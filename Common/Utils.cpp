#include "Utils.h"
#include <vector>

namespace Common {

    // --- String Conversions ---

    std::wstring Utf8ToWide(const std::string& s) {
        if (s.empty()) return L"";
        int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
        if (n <= 0) return L"";
        std::wstring w(n - 1, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
        return w;
    }

    std::string WideToAnsi(const std::wstring& w) {
        if (w.empty()) return "";
        int n = WideCharToMultiByte(CP_ACP, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
        if (n <= 0) return "";
        std::string a(n - 1, '\0');
        WideCharToMultiByte(CP_ACP, 0, w.c_str(), -1, &a[0], n, nullptr, nullptr);
        return a;
    }

    std::string Utf8ToAnsi(const std::string& s) {
        return WideToAnsi(Utf8ToWide(s));
    }

    std::wstring AnsiToWide(const std::string& s) {
        if (s.empty()) return L"";
        int n = MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, nullptr, 0);
        if (n <= 0) return L"";
        std::wstring w(n - 1, L'\0');
        MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, &w[0], n);
        return w;
    }

    // --- File Operations ---

    bool FileExists(const char* filePath) {
        if (!filePath || !*filePath) return false;
        DWORD attrib = GetFileAttributesA(filePath);
        return (attrib != INVALID_FILE_ATTRIBUTES && !(attrib & FILE_ATTRIBUTE_DIRECTORY));
    }

    // --- Window Operations ---

    bool CaptureWindowRegion(
        HWND hwnd,
        int roiX,
        int roiY,
        int roiWidth,
        int roiHeight,
        cv::Mat& outputBgr
    ) {
        if (!IsWindow(hwnd) || roiWidth <= 0 || roiHeight <= 0) {
            return false;
        }

        HDC windowDeviceContext = GetDC(hwnd);
        if (windowDeviceContext == nullptr) {
            return false;
        }

        HDC memoryDeviceContext = CreateCompatibleDC(windowDeviceContext);
        if (memoryDeviceContext == nullptr) {
            ReleaseDC(hwnd, windowDeviceContext);
            return false;
        }

        HBITMAP compatibleBitmap = CreateCompatibleBitmap(windowDeviceContext, roiWidth, roiHeight);
        if (compatibleBitmap == nullptr) {
            DeleteDC(memoryDeviceContext);
            ReleaseDC(hwnd, windowDeviceContext);
            return false;
        }

        SelectObject(memoryDeviceContext, compatibleBitmap);

        if (!BitBlt(
            memoryDeviceContext,
            0,
            0,
            roiWidth,
            roiHeight,
            windowDeviceContext,
            roiX,
            roiY,
            SRCCOPY
        )) {
            DeleteObject(compatibleBitmap);
            DeleteDC(memoryDeviceContext);
            ReleaseDC(hwnd, windowDeviceContext);
            return false;
        }

        BITMAPINFOHEADER bitmapInfoHeader;
        ZeroMemory(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER));
        bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmapInfoHeader.biWidth = roiWidth;
        bitmapInfoHeader.biHeight = -roiHeight;
        bitmapInfoHeader.biPlanes = 1;
        bitmapInfoHeader.biBitCount = 32;
        bitmapInfoHeader.biCompression = BI_RGB;

        cv::Mat bgraMat(roiHeight, roiWidth, CV_8UC4);
        if (GetDIBits(
            memoryDeviceContext,
            compatibleBitmap,
            0,
            static_cast<UINT>(roiHeight),
            bgraMat.data,
            reinterpret_cast<BITMAPINFO*>(&bitmapInfoHeader),
            DIB_RGB_COLORS
        ) == 0) {
            DeleteObject(compatibleBitmap);
            DeleteDC(memoryDeviceContext);
            ReleaseDC(hwnd, windowDeviceContext);
            return false;
        }

        cv::cvtColor(bgraMat, outputBgr, cv::COLOR_BGRA2BGR);

        DeleteObject(compatibleBitmap);
        DeleteDC(memoryDeviceContext);
        ReleaseDC(hwnd, windowDeviceContext);

        return !outputBgr.empty();
    }

} // namespace Common
