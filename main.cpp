#include <windows.h>
#include <string>
#include "LlamaInterface.h"

int main()
{
    const std::string model_path = std::string(PROJECT_ROOT) + "/models/Bonsai-27B-Q1_0.gguf";
    const std::string vision_path = std::string(PROJECT_ROOT) + "/models/Bonsai-27B-mmproj-Q8_0.gguf";
    const std::string image1 = std::string(PROJECT_ROOT) + "/images/doa.jpg";
    const std::string image2 = std::string(PROJECT_ROOT) + "/images/rdr.jpg";
    const std::string vision_prompt = "What can you see in this image? Please keep it to 1 or 2 sentences";

    LlamaInterface llamaObject{model_path, 99, 2500, vision_path};

    llamaObject.AnalyzeImage(image1, vision_prompt);
    llamaObject.AnalyzeImage(image2, vision_prompt);

    return 0;
}