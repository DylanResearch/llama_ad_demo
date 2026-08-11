#pragma once
#include "llama.h"
#include "mtmd.h"
#include <string>

// Loads a Qwen3-TTS backbone + mmproj (codec/vocoder) model pair and
// synthesizes speech for arbitrary text into a WAV file on disk.
class TtsInterface
{
public:
    // speaker_ref_path may point to a non-existent file; if so, generation
    // falls back to the model's default voice instead of voice cloning.
    TtsInterface(const std::string& model_path, const std::string& mmproj_path,
                 const std::string& speaker_ref_path, int gpu_offload_layers_amount);
    ~TtsInterface();

    TtsInterface(const TtsInterface&) = delete;
    TtsInterface& operator=(const TtsInterface&) = delete;

    // Generates speech for `text` and writes a WAV file to out_path.
    // Returns true on success.
    bool GenerateWav(const std::string& text, const std::string& out_path);

private:
    llama_model* m_pModel;
    llama_context* m_pContext;
    llama_sampler* m_pSampler;
    mtmd_context* m_pMTMDContext;
    mtmd_bitmap* m_pSpeakerBitmap; // optional, may be nullptr

    std::string m_Lang;
    int m_MaxFrames;
};
