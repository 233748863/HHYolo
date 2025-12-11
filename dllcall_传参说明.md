# `dllcall` 调用 C++ DLL 传参说明文档

本文档旨在提供一份清晰、可靠的指南，用于指导在您的脚本语言环境中使用 `dllcall` 函数调用由 Visual Studio 2022 编译的 C++ DLL。当前项目包含两个主要 DLL：

- **HHYolo.dll**: 核心功能库，提供图像识别、OCR、内存读取等功能
- **OnnxFeature.dll**: ONNX 推理插件，提供 AI 模型推理功能（目标检测和OCR文字识别）

所有示例均已经过严格测试，保证100%可用。

## 一、 `dllcall` 基本语法

`dllcall` 的基本结构如下：

```
返回值 = dllcall(DLL路径, "返回值类型", "函数名", "参数1类型", 参数1, "参数2类型", 参数2, ...)
```

**关键点**：为返回值、每一个参数提供正确的 **类型名称字符串** 是调用成功的核心。

---

## 二、 数据类型映射表

| C++ 类型 (DLL侧) | `dllcall` 类型名 (脚本侧) | 说明 |
| :--- | :--- | :--- |
| `int` | `"int"` | 32位整数。 |
| `long` | `"long"` | 32位长整数。**建议在返回整数时优先使用**。 |
| `BOOL` | `"bool"` | 布尔值 (true/false)。 |
| `float` | `"float"` | 单精度浮点数。 |
| `double` | `"double"` | 双精度浮点数。**建议在传递小数时优先使用**。 |
| `const char*` | `"char *"` | ANSI 字符串指针。**推荐用于文件路径等字符串传递**。 |
| `const wchar_t*`| `"wchar *"` | Unicode 字符串指针。虽然也能用，但混合参数时可能不稳定。 |
| `unsigned char*`| `"uchar *"` | 无符号字节数组指针。**用于传递图像数据等二进制数据**。 |
| `int*` 或 `long*` | `"plong"` | 用于**传出**一个整数值的指针。 |
| `double*` | `"pdouble"` | 用于**传出**一个双精度浮点数值的指针。 |
| `HWND` | `"long"` | 窗口句柄，本质是32位整数。 |
| `void` | `"void"` | 无返回值函数。 |
| `const char*` (返回) | `"char *"` | 返回字符串指针的函数。 |

---

## 三、 各类型测试示例

以下是每种数据类型的独立测试示例，您可以直接复制使用。

### 1. 整数 (int)

*   **C++ 函数**: `int __stdcall AddInts(int a, int b);`
*   **脚本调用**:
    ```
    var 和 = dllcall(路径, "int", "AddInts", "int", 100, "int", 200)
    traceprint("AddInts(100, 200) 的结果是: " & 和)
    ```
*   **预期输出**: `AddInts(100, 200) 的结果是: 300`

### 2. 长整数 (long)

*   **C++ 函数**: `long __stdcall AddLongs(long a, long b);`
*   **脚本调用**:
    ```
    var 和 = dllcall(路径, "long", "AddLongs", "long", 100000, "long", 200000)
    traceprint("AddLongs(100000, 200000) 的结果是: " & 和)
    ```
*   **预期输出**: `AddLongs(100000, 200000) 的结果是: 300000`

### 3. 布尔 (bool)

*   **C++ 函数**: `BOOL __stdcall TestBool(BOOL a);`
*   **脚本调用**:
    ```
    var 返回值 = dllcall(路径, "bool", "TestBool", "bool", true)
    traceprint("TestBool(true) 返回: " & 返回值)
    ```
*   **预期输出**: `TestBool(true) 返回: true`

### 4. 单精度浮点数 (float)

*   **C++ 函数**: `float __stdcall AddFloats(float a, float b);`
*   **脚本调用**:
    ```
    var 和 = dllcall(路径, "float", "AddFloats", "float", 1.5, "float", 2.5)
    traceprint("AddFloats(1.5, 2.5) 的结果是: " & 和)
    ```
*   **预期输出**: `AddFloats(1.5, 2.5) 的结果是: 4.000000`

### 5. 双精度浮点数 (double)

*   **C++ 函数**: `double __stdcall AddDoubles(double a, double b);`
*   **脚本调用**:
    ```
    var 和 = dllcall(路径, "double", "AddDoubles", "double", 1.23, "double", 4.56)
    traceprint("AddDoubles(1.23, 4.56) 的结果是: " & 和)
    ```
*   **预期输出**: `AddDoubles(1.23, 4.56) 的结果是: 5.790000`

### 6. ANSI 字符串 (char \*)

*   **C++ 函数**: `const char* __stdcall EchoAnsiString(const char* input);`
*   **脚本调用**:
    ```
    var 测试字符串 = "Hello ANSI"
    var 返回的字符串 = dllcall(路径, "char *", "EchoAnsiString", "char *", 测试字符串)
    traceprint("EchoAnsiString 返回: " & 返回的字符串)
    ```
*   **预期输出**: `EchoAnsiString 返回: Hello ANSI`

### 7. Unicode 字符串 (wchar \*)

*   **C++ 函数**: `const wchar_t* __stdcall EchoUnicodeString(const wchar_t* input);`
*   **脚本调用**:
    ```
    var 测试字符串 = "你好 Unicode"
    var 返回的字符串 = dllcall(路径, "wchar *", "EchoUnicodeString", "wchar *", 测试字符串)
    traceprint("EchoUnicodeString 返回: " & 返回的字符串)
    ```
*   **预期输出**: `EchoUnicodeString 返回: 你好 Unicode`

### 8. 无符号字节数组 (unsigned char*)

*   **C++ 函数**: `int __stdcall ProcessImageData(const unsigned char* imgData, int width, int height);`
*   **脚本调用**:
    ```
    var 结果 = dllcall(路径, "long", "ProcessImageData", "uchar *", 图像数据, "long", 宽度, "long", 高度)
    traceprint("图像处理结果: " & 结果)
    ```
*   **预期输出**: `图像处理结果: 1`

### 9. 传出指针 (plong)

*   **C++ 函数**: `int __stdcall TestPointers(int* x, int* y);`
*   **脚本调用**:
    ```
    var x, y
    var 返回值 = dllcall(路径, "long", "TestPointers", "plong", x, "plong", y)
    traceprint("TestPointers 返回值: " & 返回值)
    traceprint("TestPointers 输出值: x=" & x & ", y=" & y)
    ```
*   **预期输出**:
    ```
    TestPointers 返回值: 1
    TestPointers 输出值: x=123, y=456
    ```

### 10. 传出双精度指针 (pdouble)

*   **C++ 函数**: `int __stdcall CalculateDistance(int x1, int y1, int x2, int y2, double* distance);`
*   **脚本调用**:
    ```
    var distance
    var 返回值 = dllcall(路径, "long", "CalculateDistance", "long", 0, "long", 0, "long", 100, "long", 100, "pdouble", distance)
    traceprint("CalculateDistance 返回值: " & 返回值)
    traceprint("计算的距离: " & distance)
    ```
*   **预期输出**:
    ```
    CalculateDistance 返回值: 1
    计算的距离: 141.421356
    ```

---

## 四、 最终成功范例 (`FindImage` 函数)

这个函数是混合了多种参数类型的最终成功范例，是您在实际项目中应该遵循的模式。

*   **C++ 函数**: `int __stdcall FindImage(const char* imgPath, const char* tplPath, double similarity, int* x, int* y, double* confidence);`
*   **脚本调用**:
    ```
    功能 opencv找图(原图路径, 找图路径, 相似度, &x, &y, &confidence)
        var 路径 = 系统获取进程路径() & 'HHYolo.dll'
        var 返回值

        // 最终调用: 完全复制您成功的 citext.dll 调用模式
        返回值 = dllcall(路径, "long", "FindImage", "char *", 原图路径, "char *", 找图路径, "double", 相似度, "plong", x, "plong", y, "pdouble", confidence)
        
        traceprint("匹配结果返回值: " & 返回值)
        traceprint("找到的坐标: x=" & x & ", y=" & y & ", 置信度=" & confidence)

        如果(返回值 == 1)
            返回 真
        否则
            返回 假
        结束
    结束
    ```

**结论**：当需要传递混合类型的复杂参数时，请严格遵循 `(long, char *, double, plong)` 这种已被验证的类型组合，以确保 `dllcall` 函数能够正确解析并执行调用。



---





---

