#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>

class ClassificationPostProcessPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    ClassificationPostProcessPlugin() {}
    ClassificationPostProcessPlugin(const void* data, size_t length) {}

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override { return 1;}

    nvinfer1::DimsExprs getOutputDimensions(int, const nvinfer1::DimsExprs* inputs, int, nvinfer1::IExprBuilder& builder) noexcept override;
 

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }

    nvinfer1::DataType getOutputDataType(int, const nvinfer1::DataType* inputTypes, int) const noexcept override;


    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override;

    int initialize() noexcept override { return 0;}
    void terminate() noexcept override {}

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int, const nvinfer1::PluginTensorDesc*, int) const noexcept override
    {
        return 0;
    }

    void destroy() noexcept override { delete this;}

    size_t getSerializationSize() const noexcept override { return 0; }

    void serialize(void* buffer) const noexcept override {}

    void setPluginNamespace(const char* pluginNamespace) noexcept override
    {
        m_strNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return m_strNamespace.c_str();
    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int32_t, const nvinfer1::DynamicPluginTensorDesc*, int32_t) noexcept override;
 

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        return new ClassificationPostProcessPlugin();
    }
private:
    std::string m_strNamespace;
};

class ClassificationPostProcessPluginCreator : public nvinfer1::IPluginCreator {
public:
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &mFC;
    }

    nvinfer1::IPluginV2* createPlugin(
        const char*,
        const nvinfer1::PluginFieldCollection*) noexcept override
    {
        return new ClassificationPostProcessPlugin();
    }

    nvinfer1::IPluginV2* deserializePlugin(
        const char*,
        const void* serialData,
        size_t serialLength) noexcept override
    {
        return new ClassificationPostProcessPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) noexcept override {
        m_strNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return m_strNamespace.c_str();
    }
private:
    std::string m_strNamespace;
    nvinfer1::PluginFieldCollection mFC;
};

REGISTER_TENSORRT_PLUGIN(ClassificationPostProcessPluginCreator);

//-----//
//----//

class ClassificationPreProcessPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    ClassificationPreProcessPlugin(int outH, int outW, const float* mean, const float* std);
    ClassificationPreProcessPlugin(const void* data, size_t length);

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override { return 1;}

    nvinfer1::DimsExprs getOutputDimensions(int, const nvinfer1::DimsExprs* inputs, int, nvinfer1::IExprBuilder& builder) noexcept override;
 
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    nvinfer1::DataType getOutputDataType(int, const nvinfer1::DataType* inputTypes, int) const noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override;

    int initialize() noexcept override { return 0;}
    void terminate() noexcept override {}

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int, const nvinfer1::PluginTensorDesc*, int) const noexcept override
    {
        return 0;
    }

    void destroy() noexcept override { delete this;}

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override
    {
        m_strNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return m_strNamespace.c_str();
    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int32_t, const nvinfer1::DynamicPluginTensorDesc*, int32_t) noexcept override;
 
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override 
    {
        return new ClassificationPreProcessPlugin(m_nOutH, m_nOutW, m_fMean, m_fStd);
    }

private:
    int m_nOutH, m_nOutW;
    float m_fMean[3];
    float m_fStd[3];
    std::string m_strNamespace;
};

class ClassificationPreProcessPluginCreator : public nvinfer1::IPluginCreator {
public:
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &mFC;
    }

    nvinfer1::IPluginV2* createPlugin(
        const char*,
        const nvinfer1::PluginFieldCollection*) noexcept override
    {
        float mean[3] = {0.485f, 0.456f, 0.406f};
        float std[3]  = {0.229f, 0.224f, 0.225f};
        return new ClassificationPreProcessPlugin(240, 240, mean, std);
    }

    nvinfer1::IPluginV2* deserializePlugin(
        const char*,
        const void* serialData,
        size_t serialLength) noexcept override
    {
        return new ClassificationPreProcessPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) noexcept override {
        m_strNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return m_strNamespace.c_str();
    }
private:
    std::string m_strNamespace;
    nvinfer1::PluginFieldCollection mFC;
};
REGISTER_TENSORRT_PLUGIN(ClassificationPreProcessPluginCreator);

