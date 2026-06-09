#pragma once

#include <algorithm>
#include <cctype>
#include <string>

enum class llama_reex_env_flag {
    unset,
    disabled,
    enabled,
    invalid,
};

inline llama_reex_env_flag llama_reex_parse_env_flag(const char * value) {
    if (value == nullptr || value[0] == '\0') {
        return llama_reex_env_flag::unset;
    }

    std::string normalized(value);
    normalized.erase(
        normalized.begin(),
        std::find_if(normalized.begin(), normalized.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
    normalized.erase(
        std::find_if(normalized.rbegin(), normalized.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(),
        normalized.end());

    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return std::tolower(ch);
    });

    if (normalized.empty()) {
        return llama_reex_env_flag::unset;
    }

    if (normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes") {
        return llama_reex_env_flag::enabled;
    }

    if (normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no") {
        return llama_reex_env_flag::disabled;
    }

    return llama_reex_env_flag::invalid;
}

inline bool llama_reex_should_use_custom_unified_split(bool kv_unified, llama_reex_env_flag flag) {
    return kv_unified && flag == llama_reex_env_flag::enabled;
}
