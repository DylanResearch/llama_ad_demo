#include "TtsInterface.h"
#include "mtmd-helper.h"
#include <filesystem>
#include <iostream>

TtsInterface::TtsInterface(const std::string& model_path, const std::string& mmproj_path,
                            const std::string& speaker_ref_path, int gpu_offload_layers_amount)
    : m_pSpeakerBitmap{nullptr},
      m_Lang{"english"},
      m_MaxFrames{512}
{
    // Model
    llama_model_params model_params{llama_model_default_params()};
    model_params.n_gpu_layers = gpu_offload_layers_amount;

    m_pModel = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!m_pModel)
        std::cerr << "FAILED TO LOAD TTS MODEL: " << model_path << "\n";

    // Context - embeddings must be enabled so gen_audio can read backbone hidden states
    llama_context_params context_params{llama_context_default_params()};
    context_params.n_ctx = 8192;
    context_params.n_batch = 2048;
    context_params.embeddings = true;

    m_pContext = llama_init_from_model(m_pModel, context_params);
    if (!m_pContext)
        std::cerr << "FAILED TO LOAD TTS CONTEXT FROM MODEL\n";

    // MTMD (codec/vocoder)
    mtmd_context_params mtmd_params{mtmd_context_params_default()};
    mtmd_params.use_gpu = true;
    mtmd_params.print_timings = false;

    m_pMTMDContext = mtmd_init_from_file(mmproj_path.c_str(), m_pModel, mtmd_params);
    if (m_pMTMDContext == nullptr)
        std::cerr << "FAILED TO INIT TTS MMPROJ CONTEXT: " << mmproj_path << "\n";
    else if (mtmd_gen_audio_get_info(m_pMTMDContext).type == MTMD_GEN_AUDIO_TYPE_NONE)
        std::cerr << "MMPROJ DOES NOT SUPPORT AUDIO GENERATION: " << mmproj_path << "\n";

    // Optional speaker reference for voice cloning
    if (std::filesystem::exists(speaker_ref_path))
    {
        mtmd_helper_bitmap_wrapper wrapper{mtmd_helper_bitmap_init_from_file(m_pMTMDContext, speaker_ref_path.c_str(), false)};
        if (wrapper.bitmap)
            m_pSpeakerBitmap = wrapper.bitmap;
        else
            std::cerr << "FAILED TO LOAD TTS SPEAKER REFERENCE: " << speaker_ref_path << "\n";
    }

    // Sampling - mirrors llama-tts CLI defaults (common/arg.cpp TTS example params)
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m_pModel));
    m_pSampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_pSampler, llama_sampler_init_penalties(n_vocab, -1, 1.05f, 0.0f, 0.0f));
    llama_sampler_chain_add(m_pSampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(m_pSampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(m_pSampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(m_pSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

TtsInterface::~TtsInterface()
{
    llama_sampler_free(m_pSampler);
    if (m_pSpeakerBitmap)
        mtmd_bitmap_free(m_pSpeakerBitmap);
    if (m_pMTMDContext)
        mtmd_free(m_pMTMDContext);
    llama_free(m_pContext);
    llama_free_model(m_pModel);
}

bool TtsInterface::GenerateWav(const std::string& text, const std::string& out_path)
{
    if (!m_pMTMDContext || !m_pContext)
    {
        std::cerr << "TTS NOT INITIALIZED, SKIPPING GENERATION\n";
        return false;
    }

    // Clear seq 0's KV cache before each generation; otherwise the second call's positions
    // collide with tokens left behind by the previous generation and decode fails.
    llama_memory_seq_rm(llama_get_memory(m_pContext), 0, -1, -1);

    mtmd_helper::gen_audio gen(m_pContext, m_pMTMDContext);

    mtmd_helper_gen_audio_inp inp{};
    inp.seq_id = 0;
    inp.prompt = text.c_str();
    inp.prompt_len = text.size();
    inp.speaker_ref = m_pSpeakerBitmap;
    inp.lang = m_Lang.c_str();
    inp.top_k = 40;
    inp.top_p = 0.95f;
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;

    if (gen.set_input(&inp) != 0)
    {
        std::cerr << "TTS set_input FAILED\n";
        return false;
    }

    for (;;)
    {
        int32_t ret = gen.step_prompt(llama_n_batch(m_pContext));
        if (ret < 0)
        {
            std::cerr << "TTS PROMPT PROCESSING FAILED\n";
            return false;
        }
        if (ret == 0)
            break;
    }

    const llama_vocab* vocab = llama_model_get_vocab(m_pModel);

    auto sample_semantic_code = [&]() -> llama_token
    {
        llama_token t = llama_sampler_sample(m_pSampler, m_pContext, -1);
        llama_sampler_accept(m_pSampler, t);
        return t;
    };

    llama_token sampled = sample_semantic_code();
    const float* h_state = llama_get_embeddings_ith(m_pContext, -1);

    for (int frame = 0; frame < m_MaxFrames && !llama_vocab_is_eog(vocab, sampled); frame++)
    {
        const float* h_next = nullptr;
        if (gen.step_gen(sampled, h_state, &h_next) != 0)
        {
            std::cerr << "TTS step_gen FAILED at frame " << frame << "\n";
            return false;
        }
        h_state = h_next;
        sampled = sample_semantic_code();
    }

    int32_t sample_rate = 0;
    const char* data = nullptr;
    size_t data_len = 0;
    if (gen.get_output(&sample_rate, &data, &data_len) != 0)
    {
        std::cerr << "TTS get_output FAILED\n";
        return false;
    }

    std::filesystem::path outPath{out_path};
    std::error_code ec;
    if (outPath.has_parent_path())
        std::filesystem::create_directories(outPath.parent_path(), ec);

    FILE* f = fopen(out_path.c_str(), "wb");
    if (!f)
    {
        std::cerr << "FAILED TO OPEN TTS OUTPUT FILE: " << out_path << "\n";
        return false;
    }
    fwrite(data, 1, data_len, f);
    fclose(f);

    return true;
}
