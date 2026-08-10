#include <iostream>
#include <vector>
#include "LlamaInterface.h"

int main()
{
    //adding commment
    LlamaInterface llamaObject{"C:\\Users\\Dylan\\Documents\\DAER-IOF-audivi\\local_models_cpp\\models\\Bonsai-27B-Q1_0.gguf", 99, 2500, R"(C:\Users\Dylan\Documents\DAER-IOF-audivi\local_models_cpp\models\Bonsai-27B-mmproj-Q8_0.gguf)"};
    // llamaObject.ChatLoop();
    llamaObject.AnalyzeImage("C:\\Users\\Dylan\\Documents\\DAER-IOF-audivi\\local_models_cpp\\images\\doa.jpg", "What can you see in this image? Please keep it to 1 or 2 sentences");
    llamaObject.AnalyzeImage("C:\\Users\\Dylan\\Documents\\DAER-IOF-audivi\\local_models_cpp\\images\\rdr.jpg", "What can you see in this image? Please keep it to 1 or 2 sentences");

    return 0;
}