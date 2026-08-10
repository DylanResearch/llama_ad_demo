#include "LlamaInterface.h"
#include <iostream>
#include "mtmd-helper.h"

LlamaInterface::LlamaInterface(const std::string &model_path, int gpu_offload_layers_amount, int token_predict_amount, const std::string &vision_model_path)
    : m_TokenPredictionAmount{token_predict_amount},
      m_SequenceID{}
{
    ggml_backend_load_all();

    // only print errors
    llama_log_set([](enum ggml_log_level level, const char *text, void * /* user_data */)
                  {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        } }, nullptr);

    // Model
    llama_model_params model_params{llama_model_default_params()};
    model_params.n_gpu_layers = gpu_offload_layers_amount;

    m_pModel = llama_model_load_from_file(model_path.c_str(), model_params); // Make sure to build in x64 so model fits in memory
    if (!m_pModel)
        std::cerr << "FAILED TO LOAD MODEL: " << model_path << "\n";

    m_pVocab = llama_model_get_vocab(m_pModel);

    // Context
    llama_context_params context_params{llama_context_default_params()};
    context_params.n_ctx = 262000; // max nr of token context
    context_params.n_batch = 262000;
    context_params.no_perf = false;

    m_pContext = llama_init_from_model(m_pModel, context_params);
    if (m_pModel == nullptr)
        std::cerr << "FAILED TO LOAD CONTEXT FROM MODEL\n";

    // MTMD
    mtmd_context_params mtmd_params{mtmd_context_params_default()};
    mtmd_params.use_gpu = true;
    mtmd_params.n_threads = 4;
    mtmd_params.print_timings = true;

    m_pMTMDContext = mtmd_init_from_file(vision_model_path.c_str(), m_pModel, mtmd_params);
    if (m_pMTMDContext == nullptr)
        std::cout << "Failed to init mtmd context...\n";
    if (!mtmd_support_vision(m_pMTMDContext))
        std::cout << "VISION MODEL DOES NOT SUPPORT VISION\n";

    // Sampling
    m_pSampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_pSampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(m_pSampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(m_pSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

LlamaInterface::~LlamaInterface()
{
    llama_sampler_free(m_pSampler);
    llama_free(m_pContext);
    llama_free_model(m_pModel);
}

std::vector<llama_token> LlamaInterface::TokenizePrompt(const std::string &prompt)
{
    const int prompt_tokens_amount = -llama_tokenize(m_pVocab, prompt.c_str(), prompt.length(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(prompt_tokens_amount);
    if (llama_tokenize(m_pVocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true) < 0)
    {
        std::cerr << "FAILED TO TOKENIZE PROMPT\n";
        return std::vector<llama_token>{};
    }

    return tokens;
}

void LlamaInterface::GenerateResponse(std::vector<llama_token> &tokens)
{
    // Encode, decode batch and tokens
    llama_batch batch{llama_batch_get_one(tokens.data(), tokens.size())};
    if (llama_model_has_encoder(m_pModel))
    {
        if (llama_encode(m_pContext, batch)) // returns a 1 on error, dude why?
            std::cerr << "FAILED TO ENCODE BATCH\n";

        llama_token decoder_start_token_id{llama_model_decoder_start_token(m_pModel)};
        if (decoder_start_token_id == LLAMA_TOKEN_NULL)
            decoder_start_token_id = llama_token_bos(m_pVocab);

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    llama_token new_token_id{};

    for (int pos{0}; pos + batch.n_tokens < tokens.size() + m_TokenPredictionAmount;)
    {
        if (llama_decode(m_pContext, batch))
            std::cerr << "FAILED DECODING BATCH\n";

        pos += batch.n_tokens;

        new_token_id = llama_sampler_sample(m_pSampler, m_pContext, -1);
        if (llama_vocab_is_eog(m_pVocab, new_token_id))
            break;

        char buffer[128];
        const int text_length{llama_token_to_piece(m_pVocab, new_token_id, buffer, sizeof(buffer), 0, true)};
        if (text_length < 0)
            std::cerr << "FAILED TO CONVERT TOKEN TO TEXT\n";

        std::string text{buffer, static_cast<uint64_t>(text_length)};
        std::cout << text;
        fflush(stdout); // go straight to the console so we see a text rolling effect?

        batch = llama_batch_get_one(&new_token_id, 1);
    }
}

std::string LlamaInterface::EvaluatePromptGenerateResponse(const std::string &prompt)
{
    std::string response{};
    const bool is_first{llama_memory_seq_pos_max(llama_get_memory(m_pContext), 0) == -1}; // KV Cache https://deepwiki.com/ggml-org/llama.cpp/3.6-memory-management-and-kv-cache

    std::vector<llama_token> tokens = TokenizePrompt(prompt);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_token new_token_id{};

    while (true)
    {
        // Check if context is big enough to evaluate current batch
        uint32_t context_max{llama_n_ctx(m_pContext)};
        int context_used{llama_memory_seq_pos_max(llama_get_memory(m_pContext), 0) + 1}; // seq_pos is 0-based, to get count +1

        if (context_used + batch.n_tokens > context_max)
        {
            std::cout << "Maximum context amount was exceeded, returning early...\n";
            return std::string{"FAILED"};
        }

        // Decode batch
        if (llama_decode(m_pContext, batch)) // If this returns >0 we have a warning or fail
        {
            std::cout << "Failed to decode batch, returning early...\n";
            return std::string{"FAILED"};
        }

        // Sample next token (response)
        new_token_id = llama_sampler_sample(m_pSampler, m_pContext, -1); //-1 = last token in last batch decoded (n_tokens - 1 internally)

        // Check for end of generation
        if (llama_vocab_is_eog(m_pVocab, new_token_id))
            break; // goes to end of while loop to return response

        // Convert response tokens to string
        char buf[256];
        int text_length = llama_token_to_piece(m_pVocab, new_token_id, buf, sizeof(buf), 0, true);
        if (text_length < 0)
            std::cout << "FAILED TO CONVERT RESPONSE TOKEN TO STRING\n";

        std::string text{buf, static_cast<uint64_t>(text_length)};
        std::cout << text;
        fflush(stdout); // immediately write to console
        response += text;

        // prepare next batch
        batch = llama_batch_get_one(&new_token_id, 1); // 1 = count of tokens in array, sinze we are using autoregressive loop - array size (amount of tokens to sample) is 1
    }

    return response;
}

void LlamaInterface::ChatLoop()
{
    std::vector<llama_chat_message> messages{};
    std::vector<char> formatted(llama_n_ctx(m_pContext));
    int previous_chat_length = 0;
    const char *chat_template = llama_model_chat_template(m_pModel, "You");

    std::cout << "------------------------------------------------\n";
    std::cout << "Please write your first message to your AI slave\n";

    while (true)
    {
        // Get user input
        std::string input{};
        std::getline(std::cin, input);
        if (input.empty())
        {
            std::cout << "Quiting chat...\n";
            break; // If no input is given, exit the chat
        }

        // Add user input to messages and format
        messages.push_back({"user", input.c_str()}); // char* role, char* content
        int new_chat_length{llama_chat_apply_template(chat_template, messages.data(), messages.size(), true, formatted.data(), formatted.size())};
        if (new_chat_length > formatted.size())
        {
            formatted.resize(new_chat_length);
            new_chat_length = llama_chat_apply_template(chat_template, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        if (new_chat_length < 0)
        {
            std::cout << "Failed to apply chat template...";
            return;
        }

        // Remove previous messages to get prompt
        std::string prompt{formatted.begin() + previous_chat_length, formatted.begin() + new_chat_length};

        // Generate response
        std::string response{EvaluatePromptGenerateResponse(prompt)};

        // Add response to messages
        messages.push_back({"assistant", response.c_str()});
        previous_chat_length = llama_chat_apply_template(chat_template, messages.data(), messages.size(), false, nullptr, 0);
        if (previous_chat_length < 0)
        {
            std::cout << "Failed to apply chat template...\n";
            return;
        }

        std::cout << '\n'; // Ask for input on a new line
    }

    // free messages
    for (auto &msg : messages)
    {
        free(const_cast<char *>(msg.content));
    }
}

std::string LlamaInterface::AnalyzeImage(const std::string &image_filepath, const std::string &prompt)
{
    // Reset memory
    llama_memory_seq_rm(llama_get_memory(m_pContext), 0, -1, -1); //-1s mean unbound, remove everything in sequence

    // Create bitmap
    mtmd_helper_bitmap_wrapper bitmap_wrapper{mtmd_helper_bitmap_init_from_file(m_pMTMDContext, image_filepath.c_str(), false)};
    mtmd_bitmap *bitmap = bitmap_wrapper.bitmap;

    // Tokenize + prompt
    std::string prompt_media = prompt + " <__media__>\n"; // insert <__media__> marker into prompt to insert bitmap chunks later
    mtmd_input_text text{};
    text.text = prompt_media.c_str();
    text.text_len = prompt_media.length();
    text.add_special = true;
    text.parse_special = true;

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const mtmd_bitmap *bitmaps[] = {bitmap};
    int result = mtmd_tokenize(m_pMTMDContext, chunks, &text, bitmaps, 1); // Bitmap gets tokenized into chunks
    if (result > 0)
        std::cout << "ERROR TOKENIZING BITMAP\n";

    // Batch & encode - put it in the model memory, so we can start sampling the generated output
    llama_pos n_past{};
    llama_pos new_n_past{};
    int batch_amount = llama_n_batch(m_pContext);
    result = mtmd_helper_eval_chunks(m_pMTMDContext, m_pContext, chunks, n_past, 0, batch_amount, true, &new_n_past);

    if (result > 0)
        std::cout << "Failed to evaluate chunks\n";
    n_past = new_n_past;

    // Sample & generate output
    // Generation loop — n_past already advanced by mtmd_helper_eval_chunks()
    std::string response{};

    for (int i = 0; i < m_TokenPredictionAmount; i++)
    {
        llama_token new_token_id = llama_sampler_sample(m_pSampler, m_pContext, -1);

        if (llama_vocab_is_eog(m_pVocab, new_token_id))
            break;

        char buffer[128];
        int text_length = llama_token_to_piece(m_pVocab, new_token_id, buffer, sizeof(buffer), 0, true);
        if (text_length < 0)
        {
            std::cerr << "FAILED TO CONVERT TOKEN TO TEXT\n";
            break;
        }

        std::string text{buffer, static_cast<size_t>(text_length)};
        std::cout << text;
        response = text;
        fflush(stdout);

        llama_batch batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(m_pContext, batch))
        {
            std::cerr << "FAILED DECODING BATCH\n";
            break;
        }
        n_past++;
    }

    m_SequenceID++; // Get ready for next sequence
    return response;
}

llama_model *LlamaInterface::GetModel() const
{
    return m_pModel;
}
llama_context *LlamaInterface::GetContext() const
{
    return m_pContext;
}
llama_sampler *LlamaInterface::GetSampler() const
{
    return m_pSampler;
}
const llama_vocab *LlamaInterface::GetVocab() const
{
    return m_pVocab;
}
int LlamaInterface ::GetTokenPredictionAmount() const
{
    return m_TokenPredictionAmount;
}