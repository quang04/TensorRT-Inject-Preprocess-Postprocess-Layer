#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <vector>

struct ClassificationResult
{
    int id;
    float score;

    ClassificationResult(){}
    ClassificationResult(int _id, float _score)
        : id(_id)
        , score(_score)
    {
    }
};

class TensorRTService {
public:

    static nvinfer1::ICudaEngine* BuildEngine(const std::string& onnxPath);
    static void SaveEngine(nvinfer1::ICudaEngine* engine, const std::string& path);
    static nvinfer1::ICudaEngine* LoadEngine(const std::string& enginePath);
    static std::vector<std::string> LoadLabels(const std::string& labelPath); 
    static std::vector<ClassificationResult> Inference(nvinfer1::ICudaEngine* engine, const unsigned char* buffer, int bufferSize, int bufferW, int bufferH);

};