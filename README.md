<img width="1372" height="1009" alt="image" src="https://github.com/user-attachments/assets/dfb47a44-c750-46a9-99fb-72858b78e462" />

# llama_ad_demo

Windows desktop demo: drag an image onto a window, a local vision LLM (via [llama.cpp](https://github.com/ggml-org/llama.cpp)) describes it, and a local TTS model speaks the description aloud.

## How it works

- Drop an image file onto the app window.
- The image is analyzed by a vision-capable LLM (Bonsai-27B + mmproj) running fully offline via llama.cpp's `mtmd` multimodal support.
- The resulting text description is synthesized to speech by a Qwen3-TTS model and saved as a WAV file in `output/`, then played back.

All inference runs locally on-device (GPU offload via CUDA) — no network calls, no cloud API.

## Project layout

```
main.cpp              # entry point: wires up LlamaInterface + TtsInterface + UI
src/
  LlamaInterface.*     # llama.cpp model loading, tokenization, vision analysis
  TtsInterface.*        # Qwen3-TTS model loading, text-to-WAV synthesis
  UI.*                   # Win32 window, drag-and-drop, worker thread, playback
llama.cpp/              # git submodule (vendored dependency, not committed)
models/                 # GGUF model weights (not committed, see below)
images/                 # sample/test images (not committed)
output/                 # generated WAV files (not committed)
```

## Requirements

- Windows, Visual Studio 2022 (MSVC)
- CMake 3.28+
- CUDA toolkit (GGML_CUDA is enabled in all presets)
- OpenSSL (Win64) — only required for the MSVC Debug preset

## Setup

1. Clone with submodules:
   ```
   git clone --recurse-submodules https://github.com/DylanResearch/llama_ad_demo.git
   ```
   (or `git submodule update --init --recursive` if already cloned)

2. Download the required GGUF models into `models/`:
   - `Bonsai-27B-Q1_0.gguf` — vision LLM
   - `Bonsai-27B-mmproj-Q8_0.gguf` — vision projector
   - `Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf` — TTS backbone
   - `mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf` — TTS codec/vocoder
   - `speaker.wav` (optional) — reference voice for cloning; falls back to the default voice if absent

3. Configure and build (using the provided CMake presets):
   ```
   cmake --preset "MSVC Release"
   cmake --build build/msvc/release
   ```

4. Run the built executable, then drag an image file onto the window.

## Notes

- Model weights, the `llama.cpp` submodule, sample images, and generated `output/*.wav` files are gitignored — bring your own models.
- GPU offload defaults to all layers (`99`) for both the vision LLM and TTS model; adjust in [main.cpp](main.cpp) if running on limited VRAM.
