#pragma once
#include <string>
#include <vector>
#include <memory>
#include <sstream>


template<typename... Args>
std::string fmt_str(const std::string& format, Args&&... args) {
    size_t size = snprintf(nullptr, 0, format.c_str(), std::forward<Args>(args)...);
    if (size <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    std::unique_ptr<char[]> buf(new char[size + 1]);
    snprintf(buf.get(), size + 1, format.c_str(), std::forward<Args>(args)...);
    return std::string(buf.get(), buf.get() + size);
}

template<typename T>
inline std::string vec2str(std::vector<T> vec)
{
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << vec[i];
        if (i != vec.size() - 1) {
            ss << ", ";
        }
    }
    ss << ")";
    return ss.str();  
}

template<typename T>
inline std::string array2str(T* arr, size_t size)
{
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < size; ++i) {
        ss << arr[i];
        if (i != size - 1) {
            ss << ", ";
        }
    }
    ss << ")";
    return ss.str();  
}