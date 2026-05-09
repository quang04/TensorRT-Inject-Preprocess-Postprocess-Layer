#include "tensorRTKernel.h"
#include <NvInfer.h>
#include "tensorRTPlugin.h"

namespace
{
    char const* const CLASSIFICATION_POSTPROCESS_PLUGIN_VERSION{"1"};
    char const* const CLASSIFICATION_POSTPROCESS_PLUGIN_NAME{"ClassificationPostProcessPlugin"};

    char const* const CLASSIFICATION_PREPROCESS_PLUGIN_VERSION{"1"};
    char const* const CLASSIFICATION_PREPROCESS_PLUGIN_NAME{"ClassificationPreProcessPlugin"};
}

const char *ClassificationPostProcessPlugin::getPluginType() const noexcept
{
    return CLASSIFICATION_POSTPROCESS_PLUGIN_NAME;
}

const char *ClassificationPostProcessPlugin::getPluginVersion() const noexcept
{
    return CLASSIFICATION_POSTPROCESS_PLUGIN_VERSION;
}

nvinfer1::DimsExprs ClassificationPostProcessPlugin::getOutputDimensions(int, const nvinfer1::DimsExprs *inputs, int, nvinfer1::IExprBuilder &builder) noexcept
{
    // 1x10
    nvinfer1::DimsExprs out;
    out.nbDims = 2;
    out.d[0] = inputs[0].d[0];
    // TOPK = 5. we store format index1 index2 index 3.....score1 score2 score3
    out.d[1] = builder.constant(5*2);
    return out;
}

nvinfer1::DataType ClassificationPostProcessPlugin::getOutputDataType(int, const nvinfer1::DataType *inputTypes, int) const noexcept
{
    return nvinfer1::DataType::kFLOAT;
}

int ClassificationPostProcessPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *, cudaStream_t stream) noexcept
{
    int n = 1;

    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
        n *= inputDesc[0].dims.d[i];

    launchTopKSoftmaxKernel(
        (float*)outputs[0],
        (const float*)inputs[0],
        n,
        stream
    );

    return 0;
}

void ClassificationPostProcessPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *, int32_t, const nvinfer1::DynamicPluginTensorDesc *, int32_t) noexcept
{
}

const char *ClassificationPostProcessPluginCreator::getPluginName() const noexcept
{
    return CLASSIFICATION_POSTPROCESS_PLUGIN_NAME;
}

const char *ClassificationPostProcessPluginCreator::getPluginVersion() const noexcept
{
    return CLASSIFICATION_POSTPROCESS_PLUGIN_VERSION;
}

//-----------------//
//-----------------//
ClassificationPreProcessPlugin::ClassificationPreProcessPlugin(int outH, int outW, const float *mean, const float *std)
    : m_nOutH(outH)
    , m_nOutW(outW)

{
    memcpy(m_fMean, mean, 3*sizeof(float));
    memcpy(m_fStd, std, 3*sizeof(float));
}

ClassificationPreProcessPlugin::ClassificationPreProcessPlugin(const void *data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    memcpy(&m_nOutH, d, sizeof(int)); d += sizeof(int);
    memcpy(&m_nOutW, d, sizeof(int)); d += sizeof(int);
    memcpy(m_fMean, d, 3 * sizeof(float)); d += 3 * sizeof(float);
    memcpy(m_fStd, d, 3 * sizeof(float));
}

const char *ClassificationPreProcessPlugin::getPluginType() const noexcept
{
    return CLASSIFICATION_PREPROCESS_PLUGIN_NAME;
}

const char *ClassificationPreProcessPlugin::getPluginVersion() const noexcept
{
    return CLASSIFICATION_PREPROCESS_PLUGIN_VERSION;
}

nvinfer1::DimsExprs ClassificationPreProcessPlugin::getOutputDimensions(int, const nvinfer1::DimsExprs *inputs, int, nvinfer1::IExprBuilder &builder) noexcept
{
    // 1x3xWxH
    nvinfer1::DimsExprs out;
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = builder.constant(3);
    out.d[2] = builder.constant(m_nOutH);
    out.d[3] = builder.constant(m_nOutW);
    return out;
}

bool ClassificationPreProcessPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs) noexcept
{
    const auto& desc = inOut[pos];

    // input image format support unsigned char
    if (pos == 0) {
        return (desc.type == nvinfer1::DataType::kUINT8 || desc.type == nvinfer1::DataType::kFLOAT) && desc.format == nvinfer1::TensorFormat::kLINEAR;
    }

    if (pos == 1) {
        return desc.type == nvinfer1::DataType::kFLOAT && desc.format == nvinfer1::TensorFormat::kLINEAR;
    }

    if (pos == 2) {
        return desc.type == nvinfer1::DataType::kFLOAT && desc.format == nvinfer1::TensorFormat::kLINEAR;
    }

    return false;
}

nvinfer1::DataType ClassificationPreProcessPlugin::getOutputDataType(int, const nvinfer1::DataType *inputTypes, int) const noexcept
{
    return nvinfer1::DataType::kFLOAT;
}

int ClassificationPreProcessPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *, cudaStream_t stream) noexcept
{
    int batch = inputDesc[0].dims.d[0];
    int inC = inputDesc[0].dims.d[1];
    int inH = inputDesc[0].dims.d[2];
    int inW = inputDesc[0].dims.d[3];

    // input must be format r1g1b1 r2g2b2.....
    launchPreprocessKernel((unsigned char*)inputs[0], (float*)outputs[0], 
    inH, inW, m_nOutH, m_nOutW,
    m_fMean[0], m_fMean[1], m_fMean[2],
    m_fStd[0], m_fStd[1], m_fStd[2], stream);

    return 0;
}

size_t ClassificationPreProcessPlugin::getSerializationSize() const noexcept
{
    return 2 * sizeof(int) + 6 * sizeof(float);
}

void ClassificationPreProcessPlugin::serialize(void *buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    memcpy(d, &m_nOutH, sizeof(int)); d += sizeof(int);
    memcpy(d, &m_nOutW, sizeof(int)); d += sizeof(int);
    memcpy(d, m_fMean, 3 * sizeof(float)); d += 3 * sizeof(float);
    memcpy(d, m_fStd, 3 * sizeof(float));
}

void ClassificationPreProcessPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *, int32_t, const nvinfer1::DynamicPluginTensorDesc *, int32_t) noexcept
{
}

const char *ClassificationPreProcessPluginCreator::getPluginName() const noexcept
{
    return CLASSIFICATION_PREPROCESS_PLUGIN_NAME;
}

const char *ClassificationPreProcessPluginCreator::getPluginVersion() const noexcept
{
    return CLASSIFICATION_PREPROCESS_PLUGIN_VERSION;
}
