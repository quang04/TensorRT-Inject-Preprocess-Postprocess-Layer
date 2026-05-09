#include <iostream>
#include <string>
#include "tensorRTService.h"
#include "tensorRTLogger.h"
#include "tensorRTPlugin.h"

namespace {
    const char* outputPluginLayerName = "output_plugin";
    const char* inputPluginLayerName = "input_plugin";
    const int TOPK = 5;
}

nvinfer1::ICudaEngine *TensorRTService::BuildEngine(const std::string &onnxPath)
{
    using namespace nvinfer1;
    IBuilder* builder = createInferBuilder(theTensorRTLogger);
    auto flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flags);

    auto parser = nvonnxparser::createParser(*network, theTensorRTLogger);

    if (!parser->parseFromFile(onnxPath.c_str(), (int)ILogger::Severity::kWARNING)) {
        std::cout << "ONNX parse failed\n";
        return nullptr;
    }

    // inject custom layer preprocess to network input
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3]  = {0.229f, 0.224f, 0.225f};
    auto modelInput = network->getInput(0);

    modelInput->setName(inputPluginLayerName);

    ITensor* inputs[] = { modelInput };
    auto pluginPreProcess = new ClassificationPreProcessPlugin(240, 240, mean, std);
    auto preprocessLayer  = network->addPluginV2(inputs, 1, *pluginPreProcess);
    auto preprocessOutput = preprocessLayer->getOutput(0);

    // chain back the input layer
    for (int i = 0; i < network->getNbLayers(); i++) {
        auto layer = network->getLayer(i);

        if (layer == preprocessLayer) continue;

        for (int j = 0; j < layer->getNbInputs(); j++) {
            if (layer->getInput(j) == modelInput) {
                layer->setInput(j, *preprocessOutput);
            }
        }
    }


    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(inputPluginLayerName, nvinfer1::OptProfileSelector::kMIN, Dims4(1,3,240,240));
    profile->setDimensions(inputPluginLayerName, nvinfer1::OptProfileSelector::kOPT, Dims4(1,3,720,1280));
    profile->setDimensions(inputPluginLayerName, nvinfer1::OptProfileSelector::kMAX, Dims4(1,3,2000,2000));


    // inject custom layer postprocess to network output
    auto modelOutput = network->getOutput(0);
    auto pluginPostProcess = new ClassificationPostProcessPlugin();
    auto outputLayer = network->addPluginV2(&modelOutput, 1, *pluginPostProcess);
    network->unmarkOutput(*modelOutput);
    auto newModelOutput = outputLayer->getOutput(0);
    newModelOutput->setName(outputPluginLayerName);
    network->markOutput(*newModelOutput);


    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1ULL << 30); // 1GB
    config->addOptimizationProfile(profile);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // clean up
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();

    return engine;
}

void TensorRTService::SaveEngine(nvinfer1::ICudaEngine *engine, const std::string &path)
{
    auto serialized = engine->serialize();
    std::ofstream file(path, std::ios::binary);
    file.write((char*)serialized->data(), serialized->size());
    serialized->destroy();
}

nvinfer1::ICudaEngine *TensorRTService::LoadEngine(const std::string &enginePath)
{
    std::ifstream file(enginePath, std::ios::binary);
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    auto runtime = nvinfer1::createInferRuntime(theTensorRTLogger);

    auto engine = runtime->deserializeCudaEngine(buffer.data(), size);

    auto totalIO = engine->getNbIOTensors();

    runtime->destroy();

    return engine;
}

std::vector<std::string> TensorRTService::LoadLabels(const std::string &labelPath)
{
    std::vector<std::string> labels;

    std::ifstream f(labelPath);
    std::string line;

    if(f.is_open())
    {
        while (std::getline(f, line))
        {
            labels.push_back(line);
        }
        
        f.close();
    }
    else
    {
        std::cout << "Unable to open label file\n";
    }
    
    return labels;
}



std::vector<ClassificationResult> TensorRTService::Inference(nvinfer1::ICudaEngine *engine, 
    const unsigned char* buffer,
    int bufferSize,
    int bufferW,
    int bufferH)
{
    auto context = engine->createExecutionContext();

    // currently 1 input 1 output
    void* buffers[2];

    int inputIndex = engine->getBindingIndex(inputPluginLayerName);
    //int inputIndex = engine->getBindingIndex("input");
    int outputIndex = engine->getBindingIndex(outputPluginLayerName);
    //int outputIndex = engine->getBindingIndex("output");

    context->setBindingDimensions(inputIndex, nvinfer1::Dims4(1, 3, bufferH, bufferW));
    size_t inputSize = 1 * bufferSize * sizeof(unsigned char);
    // result size base on custom plugin preprocess
    size_t outputSize = 1 * TOPK * 2 * sizeof(float);

    cudaMalloc(&buffers[inputIndex], inputSize);
    cudaMalloc(&buffers[outputIndex], outputSize);
    cudaMemcpy(buffers[inputIndex], buffer, inputSize, cudaMemcpyHostToDevice);

    context->executeV2(buffers);

    // copy data back to cpu for display
    std::vector<float> cpuData(TOPK * 2);
    cudaMemcpy(cpuData.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost);

    std::vector<ClassificationResult> res;
    for(int i=0; i<TOPK; i++) {
        res.emplace_back(ClassificationResult(cpuData[i], cpuData[i+TOPK]));
    }

    
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    return res;
}
