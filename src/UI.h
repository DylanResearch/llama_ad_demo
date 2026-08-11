#pragma once
#define NOMINMAX
#include <windows.h>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

class LlamaInterface;
class TtsInterface;

// Simple Win32 window that accepts a dragged-in image file and shows the
// LlamaInterface vision analysis for it. The window/message loop runs on
// whichever thread calls Run(); the actual model inference runs on a
// dedicated worker thread so drag-and-drop UI stays responsive. After each
// analysis, a spoken WAV of the result is generated via TtsInterface into
// output/ad-<timestamp>.wav, also on the worker thread.
class UI
{
public:
    UI(LlamaInterface& llamaInterface, TtsInterface& ttsInterface, const std::string& visionPrompt);
    ~UI();

    UI(const UI&) = delete;
    UI& operator=(const UI&) = delete;

    // Creates the window and runs the message loop. Blocks until the window is closed.
    void Run();

private:
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
    LRESULT HandleMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    void OnCreate(HWND hwnd);
    void OnSize(int width, int height);
    void OnFilesDropped(HDROP hDrop);
    void OnAnalysisDone(WPARAM wParam, LPARAM lParam);
    void SetPreviewImage(const std::wstring& imagePath);

    // Queues an image path for the worker thread and wakes it up.
    void QueueAnalysis(const std::string& imagePath);
    // Runs on the worker thread: waits for queued images and analyzes them one at a time.
    void WorkerLoop();

    static constexpr UINT WM_ANALYSIS_DONE = WM_APP + 1;
    static constexpr int kImageWidth = 360;
    static constexpr int kImageHeight = 300;

    LlamaInterface& m_LlamaInterface;
    TtsInterface& m_TtsInterface;
    std::string m_VisionPrompt;

    HWND m_hWnd;
    HWND m_hEditResult;
    HWND m_hStaticStatus;
    HWND m_hImageStatic;
    HBITMAP m_hCurrentBitmap;
    ULONG_PTR m_GdiplusToken;

    std::thread m_WorkerThread;
    std::mutex m_Mutex;
    std::condition_variable m_CV;
    std::string m_PendingImagePath;
    bool m_HasPending;
    bool m_Running;
};
