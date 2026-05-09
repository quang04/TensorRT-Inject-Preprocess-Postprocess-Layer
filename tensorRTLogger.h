#pragma once

#include <NvInferRuntimeCommon.h>
#include <iostream>

class TensorRTLogger : public nvinfer1::ILogger
{
public:
    explicit TensorRTLogger(nvinfer1::ILogger::Severity serverity = nvinfer1::ILogger::Severity::kWARNING)
        : m_reportSeverity(serverity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        std::cout << msg;
    }

    nvinfer1::ILogger& GetTrtLogger() noexcept
    {
        return *this;
    }

    inline nvinfer1::ILogger::Severity GetReportSeverity() const {return m_reportSeverity;}
private:
    nvinfer1::ILogger::Severity m_reportSeverity;

};

// only one logger instance
extern TensorRTLogger theTensorRTLogger;

