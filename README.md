# llama_ad_demo

Windows 桌面演示程序：将图片拖入窗口后，本地视觉 LLM（基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)）对图片进行描述，随后本地 TTS 模型将描述内容朗读出来。

## 工作原理

- 将图片文件拖放到程序窗口中。
- 图片由具备视觉能力的 LLM（Bonsai-27B + mmproj）分析，完全离线运行，依赖 llama.cpp 的 `mtmd` 多模态支持。
- 生成的文字描述由 Qwen3-TTS 模型合成语音，保存为 `output/` 目录下的 WAV 文件，并自动播放。

所有推理均在本机运行（通过 CUDA 进行 GPU 加速）——无需联网，不依赖任何云端 API。

## 项目结构

```
main.cpp              # 入口文件：组装 LlamaInterface + TtsInterface + UI
src/
  LlamaInterface.*     # llama.cpp 模型加载、分词、视觉分析
  TtsInterface.*        # Qwen3-TTS 模型加载、文本转 WAV 合成
  UI.*                   # Win32 窗口、拖放、工作线程、播放
llama.cpp/              # git 子模块（第三方依赖，不纳入版本控制）
models/                 # GGUF 模型权重文件（不纳入版本控制，见下方说明）
images/                 # 示例/测试图片（不纳入版本控制）
output/                 # 生成的 WAV 文件（不纳入版本控制）
```

## 环境要求

- Windows，Visual Studio 2022（MSVC）
- CMake 3.28 及以上
- CUDA 工具链（所有预设均启用 GGML_CUDA）
- OpenSSL（Win64）——仅 MSVC Debug 预设需要

## 安装步骤

1. 克隆仓库并包含子模块：
   ```
   git clone --recurse-submodules https://github.com/DylanResearch/llama_ad_demo.git
   ```
   （若已克隆，可执行 `git submodule update --init --recursive`）

2. 将所需的 GGUF 模型文件下载至 `models/` 目录：
   - `Bonsai-27B-Q1_0.gguf` —— 视觉 LLM
   - `Bonsai-27B-mmproj-Q8_0.gguf` —— 视觉投影模型
   - `Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf` —— TTS 主干模型
   - `mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf` —— TTS 编解码/声码器模型
   - `speaker.wav`（可选）—— 用于声音克隆的参考音频；缺失时自动使用默认音色

3. 配置并构建（使用项目提供的 CMake 预设）：
   ```
   cmake --preset "MSVC Release"
   cmake --build build/msvc/release
   ```

4. 运行生成的可执行文件，然后将图片文件拖入窗口。

## 说明

- 模型权重、`llama.cpp` 子模块、示例图片以及生成的 `output/*.wav` 文件均已在 `.gitignore` 中排除——请自行准备模型文件。
- 视觉 LLM 与 TTS 模型的 GPU 加载层数默认均为全部层（`99`）；若显存有限，请在 [main.cpp](main.cpp) 中调整该参数。
