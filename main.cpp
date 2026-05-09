#include "plugin/tensorRTPlugin.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include "tensorRTLogger.h"
#include "tensorRTService.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

TensorRTLogger theTensorRTLogger{TensorRTLogger::Severity::kINFO};

namespace {
    const char* const ONNX_MODEL_PATH =  R"(E:\Me\Github\TensorRT\src\efficientnet_b1.onnx)";
    const char* const TENSORRT_MODEL_PATH  = R"(E:\Me\Github\TensorRT\src\model_postprocess_preprocess.opt)";
    const char* const IMAGE_TESTING =  R"(E:\Me\Github\TensorRT\src\volcano.jpg)";
    const char* const LABEL_PATH = R"(E:\Me\Github\TensorRT\src\label.txt)";
}


int main(int, char**){
    
    std::filesystem::path p(TENSORRT_MODEL_PATH);
    if(!std::filesystem::exists(p))
    {
        std::string onnxModelPath = ONNX_MODEL_PATH;
        auto onnxParser = TensorRTService::BuildEngine(onnxModelPath);
        if(onnxParser == nullptr)
        {
            return -1;
        }

        TensorRTService::SaveEngine(onnxParser, TENSORRT_MODEL_PATH);
    }

    auto labels = TensorRTService::LoadLabels(LABEL_PATH);
    auto tensorRTEngine = TensorRTService::LoadEngine(TENSORRT_MODEL_PATH);  
    int imgW = 0, imgH = 0, imgC = 0;
    auto buffers = stbi_load(IMAGE_TESTING, &imgW, &imgH, &imgC, 0);
    
    // image format r1g1b1 r2g2b2 .....
    auto results = TensorRTService::Inference(tensorRTEngine, buffers, imgW*imgH*imgC, imgW, imgH);
    
    std::cout << "-----\n";
    for(auto const& r : results)
    {
        std::cout << "Class: " << labels[r.id] << " Score: " << r.score << std::endl;
    }

    // make sure everything complete before shutdown
    cudaDeviceSynchronize();

    return 0;
}
