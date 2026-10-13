#include <windows.h>
#include <string>
#include "LlamaInterface.h"
#include "TtsInterface.h"
#include "UI.h"
#include <fstream>
#include <iterator>

int main()
{
    auto stream = std::ifstream(std::string(PROJECT_ROOT) + "/prompt/vision_prompt.txt");
    std::string str(std::istreambuf_iterator<char>{stream}, {});

    const std::string model_path = std::string(PROJECT_ROOT) + "/models/Bonsai-27B-Q1_0.gguf";
    const std::string vision_path = std::string(PROJECT_ROOT) + "/models/Bonsai-27B-mmproj-Q8_0.gguf";
    const std::string vision_prompt = str;

    const std::string tts_model_path = std::string(PROJECT_ROOT) + "/models/Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf";
    const std::string tts_mmproj_path = std::string(PROJECT_ROOT) + "/models/mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf";
    const std::string tts_speaker_path = std::string(PROJECT_ROOT) + "/models/speaker.wav"; // optional, falls back to default voice if missing

    LlamaInterface llamaObject{model_path, 99, 2500, vision_path};
    TtsInterface ttsObject{tts_model_path, tts_mmproj_path, tts_speaker_path, 99};

    UI ui{llamaObject, ttsObject, vision_prompt};
    ui.Run();

    return 0;
}