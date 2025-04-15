#pragma once
#include<vector>
#include<algorithm>
#include<numeric>
#include<string>
#include<unordered_map>
#include<memory>
#include "src/utils/macro.h"
#include "src/utils/string_utils.h"
enum class Device
{
    CPU = 0,
    GPU = 1,
    UNDEFINED = 2,
};

enum  class DataType
{
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    BF16 = 4,
    INT32 = 5,
    Bool = 6,
    BYTES = 7,
    UNDEFINED_DTYPE = 16,
};

template<typename T>
DataType getTensorType()
{
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, const float>)
    {
        return DataType::FP32;
    }
    else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, const half>)
    {
        return DataType::FP16;
    }
    else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T const int8_t>)
    {
        return DataType::INT8;
    }
    else if constexpr (std::is_same_v<T, int4_t> || std::is_same_v<T,const int4_t>)
    {
        return DataType::INT4;
    }
    else if constexpr(std::is_same_v<T,int> || std::is_same_v<T,const int>)
    {
        return DataType::INT32;
    }
    else if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, const bool>)
    {
        return DataType::Bool;
    }
    else if constexpr (std::is_same_v<T, char> || std::is_same_v<T, const char>)
    {
        return DataType::BYTES;
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return DataType::BF16;
    }
    else
    {
        return DataType::UNDEFINED_DTYPE;
    }
}

template<typename T>
class Tensor;

struct TensorBase
{
    Device device;
    DataType dtype;
    std::vector<int> shape;
    TensorBase() = default;
    TensorBase(const Device& device, const DataType& dtype, const std::vector<int>& shape)
        : device(device), dtype(dtype), shape(shape) {}
    virtual int size() const {
        if(shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }
    template<typename T>
    Tensor<T>* cast() {
        return static_cast<Tensor<T>*>(this);
    }
    std::string deviceString() const {
        switch (device) {
            case Device::CPU: return "CPU";
            case Device::GPU: return "GPU";
            default: return "UNDEFINED";
        }
    }
    virtual std::string toString() const {
        static const std::unordered_map<DataType, std::string> dtype_map = {
            {DataType::FP32, "FP32"},
            {DataType::FP16, "FP16"},
            {DataType::INT8, "INT8"},
            {DataType::INT4, "INT4"},
            {DataType::BF16, "BF16"},
            {DataType::INT32, "INT32"},
            {DataType::Bool, "Bool"},
            {DataType::BYTES, "BYTES"},
            {DataType::UNDEFINED_DTYPE, "UNDEFINED_DTYPE"}
        };
        return fmt_str("TensorBase(device=%s, dtype=%s, shape=%s)", deviceString().c_str(),dtype_map.at(dtype).c_str(), vec2str(shape).c_str());
    }
};


template<typename T>
class Tensor: public TensorBase
{
public:
    T* data = nullptr;
    Tensor() = default;
    Tensor(const Device& device, const DataType& dtype, const std::vector<int>& shape)
        : TensorBase(device, dtype, shape) {}
    Tensor(const Device& device, const DataType& dtype, const std::vector<int>& shape, T* data)
        : TensorBase(device, dtype, shape), data(data) {}

    int size() const override {
       if(data == nullptr || shape.size()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }
    int T getVal(int index = 0) const
    {
        LLM_ASSERT(data != nullptr, "Tensor data is null");
        LLM_ASSERT(index>=0 && index<size(), "Index out of range");
        LLM_ASSERT(device==Device::CPU, "getVal() is only supported for CPU tensors");
        return data[index];
    }
    inline T* getData() const
    {
        LLM_ASSERT(data != nullptr, "Tensor data is null");
        return data;
    }

    virtual std::string toString() const override {
        return fmt_str("Tensor(device=%s, dtype=%s , shape=%s, data=%p)",
            deviceString().c_str(), 
            dataTypeString().c_str(), 
            vec2str(shape).c_str(), 
            data);
    }
    std::string dataTypeString() const {
        switch (dtype) {
            case DataType::FP32: return "FP32";
            case DataType::FP16: return "FP16";
            case DataType::INT8: return "INT8";
            case DataType::INT4: return "INT4";
            case DataType::BF16: return "BF16";
            case DataType::INT32: return "INT32";
            case DataType::Bool: return "Bool";
            case DataType::BYTES: return "BYTES";
            default: return "UNDEFINED_DTYPE";
        }
    }
};
using TensorPtr = std::shared_ptr<TensorBase>;
struct TensorMap
{
    std::unordered_map<std::string, TensorPtr> tensors;
    TensorMap() = default;
    TensorMap(std::initializer_list<std::pair<std::string,TensorPtr>> tensor_map) {
        for (const auto& pair : tensor_map) {
           insert(pair.first, pair.second);
        }
    }
    TensorMap(const std::unordered_map<std::string, TensorPtr>& tensor_map){
        for (const auto& pair : tensor_map) {
           insert(pair.first, pair.second);
        }
    }
    TensorMap(const TensorMap& other) : tensors(other.tensors) {}
   
    TensorPtr getTensor(const std::string& name) const {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            return it->second;
        }
        return nullptr;
    }

    inline void insert(const std::string& name, TensorPtr tensor) {
        tensors[name] = tensor;
    }
    inline void insert(std::pair<std::string, TensorPtr> pair) {
        tensors[pair.first] = pair.second;
    }
    bool isValid(TensorBase* tensor) const {
        return tensor != nullptr && tensor->size()>0;
    }

};



