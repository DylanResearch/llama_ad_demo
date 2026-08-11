#include "UI.h"
#include "LlamaInterface.h"
#include "TtsInterface.h"
#include <shellapi.h>
#include <objidl.h>
#include <gdiplus.h>
#include <mmsystem.h>
#include <algorithm>
#include <ctime>
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "winmm.lib")

namespace
{
    // Loads an image file and scales/letterboxes it to fit boxWidth x boxHeight, preserving aspect ratio.
    HBITMAP LoadScaledBitmap(const std::wstring& path, int boxWidth, int boxHeight)
    {
        Gdiplus::Bitmap source(path.c_str());
        if (source.GetLastStatus() != Gdiplus::Ok)
            return nullptr;

        UINT srcW = source.GetWidth();
        UINT srcH = source.GetHeight();
        if (srcW == 0 || srcH == 0)
            return nullptr;

        double scale = std::min(static_cast<double>(boxWidth) / srcW, static_cast<double>(boxHeight) / srcH);
        int destW = std::max(1, static_cast<int>(srcW * scale));
        int destH = std::max(1, static_cast<int>(srcH * scale));
        int offsetX = (boxWidth - destW) / 2;
        int offsetY = (boxHeight - destH) / 2;

        Gdiplus::Bitmap canvas(boxWidth, boxHeight, PixelFormat32bppARGB);
        Gdiplus::Graphics graphics(&canvas);
        graphics.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
        graphics.Clear(Gdiplus::Color(255, 240, 240, 240));
        graphics.DrawImage(&source, offsetX, offsetY, destW, destH);

        HBITMAP hBitmap{};
        canvas.GetHBITMAP(Gdiplus::Color(255, 240, 240, 240), &hBitmap);
        return hBitmap;
    }

    std::wstring Utf8ToWide(const std::string& utf8)
    {
        if (utf8.empty())
            return std::wstring{};

        int size = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
        std::wstring wide(size - 1, L'\0'); // size includes the null terminator
        MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, wide.data(), size);
        return wide;
    }

    std::string WideToUtf8(const std::wstring& wide)
    {
        if (wide.empty())
            return std::string{};

        int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string utf8(size - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, utf8.data(), size, nullptr, nullptr);
        return utf8;
    }

    // Timestamp string usable as a filename component, e.g. "20260811-153045".
    std::string MakeTimestamp()
    {
        std::time_t now = std::time(nullptr);
        std::tm localTime{};
        localtime_s(&localTime, &now);

        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", &localTime);
        return std::string{buffer};
    }
}

UI::UI(LlamaInterface& llamaInterface, TtsInterface& ttsInterface, const std::string& visionPrompt)
    : m_LlamaInterface{llamaInterface},
      m_TtsInterface{ttsInterface},
      m_VisionPrompt{visionPrompt},
      m_hWnd{},
      m_hEditResult{},
      m_hStaticStatus{},
      m_hImageStatic{},
      m_hCurrentBitmap{},
      m_GdiplusToken{},
      m_HasPending{false},
      m_Running{true}
{
    Gdiplus::GdiplusStartupInput gdiplusStartupInput{};
    Gdiplus::GdiplusStartup(&m_GdiplusToken, &gdiplusStartupInput, nullptr);

    m_WorkerThread = std::thread(&UI::WorkerLoop, this);
}

UI::~UI()
{
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_Running = false;
    }
    m_CV.notify_all();
    if (m_WorkerThread.joinable())
        m_WorkerThread.join();

    if (m_hCurrentBitmap)
        DeleteObject(m_hCurrentBitmap);

    Gdiplus::GdiplusShutdown(m_GdiplusToken);
}

void UI::Run()
{
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    const char* className = "AudiviUIWindowClass";

    WNDCLASS wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = className;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    RegisterClass(&wc);

    CreateWindowEx(WS_EX_ACCEPTFILES, className, "Audivi - Image Analysis",
                   WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
                   nullptr, nullptr, hInstance, this);

    ShowWindow(m_hWnd, SW_SHOW);

    MSG msg{};
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

LRESULT CALLBACK UI::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    UI* pThis{};
    if (msg == WM_NCCREATE)
    {
        CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
        pThis = reinterpret_cast<UI*>(pCreate->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
        pThis->m_hWnd = hwnd;
    }
    else
    {
        pThis = reinterpret_cast<UI*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    }

    if (pThis)
        return pThis->HandleMessage(hwnd, msg, wParam, lParam);

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

LRESULT UI::HandleMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CREATE:
        OnCreate(hwnd);
        return 0;

    case WM_SIZE:
        OnSize(LOWORD(lParam), HIWORD(lParam));
        return 0;

    case WM_DROPFILES:
    {
        HDROP hDrop = reinterpret_cast<HDROP>(wParam);
        OnFilesDropped(hDrop);
        DragFinish(hDrop);
        return 0;
    }

    case WM_ANALYSIS_DONE:
        OnAnalysisDone(wParam, lParam);
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

void UI::OnCreate(HWND hwnd)
{
    m_hStaticStatus = CreateWindowW(L"STATIC", L"Drag an image file onto this window to analyze it.",
                                     WS_CHILD | WS_VISIBLE,
                                     10, 10, 760, 20, hwnd, nullptr, GetModuleHandle(nullptr), nullptr);

    m_hImageStatic = CreateWindowW(L"STATIC", L"",
                                    WS_CHILD | WS_VISIBLE | SS_BITMAP | SS_CENTERIMAGE | SS_SUNKEN,
                                    10, 40, kImageWidth, kImageHeight, hwnd, nullptr, GetModuleHandle(nullptr), nullptr);

    m_hEditResult = CreateWindowW(L"EDIT", L"",
                                   WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL,
                                   10 + kImageWidth + 10, 40, 400, kImageHeight, hwnd, nullptr, GetModuleHandle(nullptr), nullptr);

    HFONT hFont = reinterpret_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
    SendMessage(m_hStaticStatus, WM_SETFONT, reinterpret_cast<WPARAM>(hFont), TRUE);
    SendMessage(m_hEditResult, WM_SETFONT, reinterpret_cast<WPARAM>(hFont), TRUE);
}

void UI::OnSize(int width, int height)
{
    if (!m_hEditResult || !m_hStaticStatus || !m_hImageStatic)
        return;

    const int margin = 10;
    const int imageTop = 40;

    MoveWindow(m_hStaticStatus, margin, margin, width - 2 * margin, 20, TRUE);
    MoveWindow(m_hImageStatic, margin, imageTop, kImageWidth, kImageHeight, TRUE);

    int editLeft = margin + kImageWidth + margin;
    int editWidth = width - editLeft - margin;
    if (editWidth < 50)
        editWidth = 50;
    int editHeight = height - imageTop - margin;
    if (editHeight < 50)
        editHeight = 50;
    MoveWindow(m_hEditResult, editLeft, imageTop, editWidth, editHeight, TRUE);
}

void UI::OnFilesDropped(HDROP hDrop)
{
    UINT count = DragQueryFileW(hDrop, 0xFFFFFFFF, nullptr, 0);
    if (count == 0)
        return;

    wchar_t pathBuffer[MAX_PATH];
    if (!DragQueryFileW(hDrop, 0, pathBuffer, MAX_PATH))
        return;

    std::wstring statusText = L"Analyzing: " + std::wstring(pathBuffer);
    SetWindowTextW(m_hStaticStatus, statusText.c_str());
    SetWindowTextW(m_hEditResult, L"");

    SetPreviewImage(pathBuffer);
    QueueAnalysis(WideToUtf8(pathBuffer));
}

void UI::SetPreviewImage(const std::wstring& imagePath)
{
    HBITMAP hBitmap = LoadScaledBitmap(imagePath, kImageWidth, kImageHeight);
    if (!hBitmap)
        return;

    SendMessage(m_hImageStatic, STM_SETIMAGE, IMAGE_BITMAP, reinterpret_cast<LPARAM>(hBitmap));

    if (m_hCurrentBitmap)
        DeleteObject(m_hCurrentBitmap);
    m_hCurrentBitmap = hBitmap;
}

void UI::OnAnalysisDone(WPARAM wParam, LPARAM lParam)
{
    std::wstring* pResult = reinterpret_cast<std::wstring*>(lParam);
    std::wstring* pStatus = reinterpret_cast<std::wstring*>(wParam);
    SetWindowTextW(m_hEditResult, pResult->c_str());
    SetWindowTextW(m_hStaticStatus, pStatus->c_str());
    delete pResult;
    delete pStatus;
}

void UI::QueueAnalysis(const std::string& imagePath)
{
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_PendingImagePath = imagePath;
        m_HasPending = true;
    }
    m_CV.notify_one();
}

void UI::WorkerLoop()
{
    while (true)
    {
        std::string imagePath;
        {
            std::unique_lock<std::mutex> lock(m_Mutex);
            m_CV.wait(lock, [this] { return m_HasPending || !m_Running; });
            if (!m_Running)
                return;

            imagePath = m_PendingImagePath;
            m_HasPending = false;
        }

        std::string result = m_LlamaInterface.AnalyzeImage(imagePath, m_VisionPrompt);

        std::string wavPath = std::string(PROJECT_ROOT) + "/output/ad-" + MakeTimestamp() + ".wav";
        bool wavOk = m_TtsInterface.GenerateWav(result, wavPath);

        if (wavOk)
            PlaySoundA(wavPath.c_str(), nullptr, SND_FILENAME | SND_ASYNC);

        // Marshal the result over to the UI thread; ownership transfers to OnAnalysisDone.
        std::wstring statusSuffix = wavOk
            ? L"Ready - wav saved to " + Utf8ToWide(wavPath)
            : L"Ready - wav generation failed.";
        std::wstring* pResult = new std::wstring(Utf8ToWide(result));
        std::wstring* pStatus = new std::wstring(statusSuffix);
        PostMessage(m_hWnd, WM_ANALYSIS_DONE, reinterpret_cast<WPARAM>(pStatus), reinterpret_cast<LPARAM>(pResult));
    }
}
