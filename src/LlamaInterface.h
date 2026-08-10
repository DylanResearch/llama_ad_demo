#include "llama.h"
#include <string>
#include <vector>
#include "mtmd.h"

class LlamaInterface
{
public:
    LlamaInterface(const std::string &model_path, int gpu_offload_layers_amount, int token_predict_amount, const std::string& vision_model_path = "");
    ~LlamaInterface();

    std::vector<llama_token> TokenizePrompt(const std::string &prompt);
    void GenerateResponse(std::vector<llama_token> &tokens);
    
    std::string EvaluatePromptGenerateResponse(const std::string& prompt);
    void ChatLoop();

    std::string AnalyzeImage(const std::string& image_filepath, const std::string& prompt);

    //Getters
    llama_model* GetModel() const;
    llama_context* GetContext() const;
    llama_sampler* GetSampler() const;
    const llama_vocab* GetVocab() const;
    int GetTokenPredictionAmount() const;

private:
    llama_model *m_pModel;
    llama_context *m_pContext;
    llama_sampler *m_pSampler;
    const llama_vocab *m_pVocab;

    mtmd_context* m_pMTMDContext;

    int m_TokenPredictionAmount;
    int m_SequenceID;
};