#pragma once
#include <windows.h>

// 前向声明 cv::Mat
namespace cv { class Mat; }

#ifdef HHYOLO_EXPORTS
#define HHAPI __declspec(dllexport)
#else
#define HHAPI __declspec(dllimport)
#endif




extern "C" {

/**
 * @brief 在主图中查找模板图的位置 (文件路径版)。
 * @param imgPath 主图像的ANSI路径。
 * @param tplPath 模板图像的ANSI路径。
 * @param similarity 相似度阈值 (0.0 ~ 1.0)。
 * @param x 用于接收匹配位置左上角X坐标的指针。
 * @param y 用于接收匹配位置左上角Y坐标的指针。
 * @return 1 表示成功找到匹配，0 表示未找到，-1 表示发生内部错误。
 */
HHAPI int __stdcall FindImage(
    const char* imgPath,
    const char* tplPath,
    double similarity,
    int* x,
    int* y,
    double* confidence);

/**
 * @brief 在内存中的主图中查找模板图的位置 (内存版)。
 * @param imgData 主图像的像素数据 (通常是BGR或BGRA格式)。
 * @param imgWidth 主图像的宽度。
 * @param imgHeight 主图像的高度。
 * @param imgChannels 主图像的通道数 (3 for BGR, 4 for BGRA)。
 * @param tplPath 模板图像的ANSI路径。
 * @param similarity 相似度阈值 (0.0 ~ 1.0)。
 * @param x 用于接收匹配位置左上角X坐标的指针。
 * @param y 用于接收匹配位置左上角Y坐标的指针。
 * @return 1 表示成功找到匹配，0 表示未找到，-1 表示发生内部错误。
 */
HHAPI int __stdcall FindImageFromMem(
    const unsigned char* imgData,
    int imgWidth,
    int imgHeight,
    int imgChannels,
    const char* tplPath,
    double similarity,
    int* x,
    int* y,
    double* confidence);

/**
 * @brief 对指定窗口句柄截图并在截图中查找模板图。
 * @param hwnd 目标窗口的句柄。
 * @param roiX 截图区域左上角的X坐标 (相对于窗口客户区)。
 * @param roiY 截图区域左上角的Y坐标 (相对于窗口客户区)。
 * @param roiWidth 截图区域的宽度。
 * @param roiHeight 截图区域的高度。
 * @param tplPath 模板图像的ANSI路径。
 * @param similarity 相似度阈值 (0.0 ~ 1.0)。
 * @param matchX 用于接收匹配位置左上角X坐标的指针 (坐标相对于窗口客户区)。
 * @param matchY 用于接收匹配位置左上-角Y坐标的指针 (坐标相对于窗口客户区)。
 * @return 1=找到, 0=未找到, -1=OpenCV内部错误, -2=无效参数, -3=GDI对象创建失败, -4=截图失败, -5=内存分配失败。
 */
HHAPI int __stdcall CaptureAndFindImage(
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    const char* tplPath,
    double similarity,
    int* matchX,
    int* matchY,
    double* confidence);

/**
 * @brief 对指定窗口截图并查找所有匹配的模板图。
 * @param hwnd 目标窗口的句柄。
 * @param roiX 截图区域左上角的X坐标 (相对于窗口客户区)。
 * @param roiY 截图区域左上角的Y坐标 (相对于窗口客户区)。
 * @param roiWidth 截图区域的宽度。
 * @param roiHeight 截图区域的高度。
 * @param tplPath 模板图像的ANSI路径。
 * @param similarity 相似度阈值 (0.0 ~ 1.0)。
 * @return 成功时返回"数量;x1,y1|x2,y2..."格式的字符串, 未找到返回"0;", 失败返回"-1"。
 */
HHAPI const char* __stdcall CaptureAndFindAllImages(
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    const char* tplPath,
    double similarity
);

/**
 * @brief 根据地址表达式读取目标进程的内存。
 * @param hwnd 目标窗口的句柄。
 * @param addressExpression 地址表达式，格式如 "[module.exe+offset]+offset1+offset2"。
 * @param readType 读取类型 ("i", "f", "d", "s", "w")。
 * @param readSize 读取的字节数。
 * @return 成功时返回包含结果的字符串；失败时返回字符串"-1"。
 */
HHAPI const char* __stdcall ReadMemory(
    HWND hwnd,
    const char* addressExpression,
    const char* readType,
    int readSize
);

/**
 * @brief 在指定窗口的ROI区域内截图，并使用多个模板图片同时进行搜索。
 * @param hwnd 目标窗口的句柄。
 * @param roiX, roiY, roiWidth, roiHeight 截图区域。
 * @param multiTplPaths 多个模板图片的路径，用'|'分隔。
 * @param similarity 相似度阈值。
 * @return 成功时返回"文件名1,x,y,置信度|文件名2,x,y,置信度..."格式的字符串, 未找到返回"", 失败返回"-1"。
 */
HHAPI const char* __stdcall CaptureAndFindMultiTemplates(
    HWND hwnd,
    int roiX, int roiY, int roiWidth, int roiHeight,
    const char* multiTplPaths,
    double similarity
);

/**
 * @brief 在指定坐标获取颜色，并与颜色列表进行匹配。
 * @param hwnd 目标窗口的句柄。如果为0，则x,y为屏幕绝对坐标；否则为相对坐标。
 * @param x 要检测的X坐标。
 * @param y 要检测的Y坐标。
 * @param colorStr 颜色匹配字符串，格式为 "RRGGBB-偏色值|RRGGBB-偏色值|..."。
 * @return 成功找到匹配时返回"1|3|..."格式的索引字符串；未找到返回"0"；失败返回"-1"。
 */
HHAPI const char* __stdcall FindColor(
    HWND hwnd,
    int x,
    int y,
    const char* colorStr
);


/**
 * @brief 通过定位“锚点”图片，并基于固定的网格布局参数，批量分析网格中各个单元格的状态。
 * @details 这是一个为固定UI布局设计的、高性能的批量分析API。它首先在窗口截图中寻找锚点，然后根据传入的几何参数推算出每个格子的位置，并分析每个格子内部的状态（例如血条）。
 * @param hwnd 目标窗口句柄。
 * @param anchorTplPath 用于定位整个框架的锚点模板图片路径。
 * @param similarity 查找锚点的相似度阈值 (0.0 ~ 1.0)。
 * @param gridRows 网格的行数。
 * @param gridCols 网格的列数。
 * @param firstCellOffsetX 从锚点左上角到第一个格子左上角的X方向距离（像素）。
 * @param firstCellOffsetY 从锚点左上角到第一个格子左上角的Y方向距离（像素）。
 * @param cellWidth 单个格子的完整宽度。
 * @param cellHeight 单个格子的完整高度。
 * @param horizontalGap 相邻格子之间的水平间距。
 * @param verticalGap 相邻格子之间的垂直间距。
 * @param healthBarOffsetX 从格子左上角到其内部血条区域左上角的X方向距离。
 * @param healthBarOffsetY 从格子左上角到其内部血条区域左上角的Y方向距离。
 * @param healthBarWidth 血条区域本身的宽度。
 * @param healthBarHeight 血条区域本身的高度。
 * @return const char* 格式化的结果字符串: "centerX1,centerY1,health1|centerX2,centerY2,health2|..."。
 *         如果锚点未找到，返回 "-1"。
 *         如果发生截图失败等其他错误，返回 "-2"。
 */
HHAPI const char* __stdcall AnalyzeGrid(
    HWND hwnd,
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
);

/**
 * @brief 对指定区域进行OCR文本识别。
 * @param hwnd 目标窗口的句柄。
 * @param roiX, roiY, roiWidth, roiHeight 要进行文字识别的区域 (ROI)，坐标相对于窗口客户区。
 * @param lang 识别语言，如 "chi_sim"。
 * @param tessdataPath 包含 .traineddata 语言包的文件夹路径。
 * @return 成功时返回 "文本1,x1,y1,w1,h1|文本2,x2,y2,w2,h2|..." 格式的字符串，坐标为相对于ROI的坐标。
 *         未识别到文本返回 ""。
 *         失败返回 "-1"。
 */
HHAPI const char* __stdcall Ocr(
    HWND hwnd,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    const char* lang,
    const char* tessdataPath
);



/**
 * @brief 在指定区域内查找特定文本，并返回其中心坐标。
 * @param hwnd 目标窗口的句柄。
 * @param roiX, roiY, roiWidth, roiHeight 要进行文字识别的区域 (ROI)，坐标相对于窗口客户区。
 * @param textToFind 要查找的文本字符串 (ANSI 编码)。
 * @param lang 识别语言，如 "chi_sim"。
 * @param tessdataPath 包含 .traineddata 语言包的文件夹路径。
 * @return 成功时返回 "x,y" 格式的中心坐标字符串, 未找到返回 "0", 发生错误返回 "-1"。
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
);

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
HHAPI const char* __stdcall OcrFile(
    const char* imgPath,
    const char* lang,
    const char* tessdataPath
);


} // extern "C"