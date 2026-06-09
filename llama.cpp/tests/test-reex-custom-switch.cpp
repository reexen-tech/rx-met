#include "../src/llama-reex-custom.h"

#include <cassert>

int main() {
    assert(llama_reex_parse_env_flag(nullptr) == llama_reex_env_flag::unset);
    assert(llama_reex_parse_env_flag("") == llama_reex_env_flag::unset);
    assert(llama_reex_parse_env_flag(" 1 ") == llama_reex_env_flag::enabled);
    assert(llama_reex_parse_env_flag("true") == llama_reex_env_flag::enabled);
    assert(llama_reex_parse_env_flag("ON") == llama_reex_env_flag::enabled);
    assert(llama_reex_parse_env_flag("0") == llama_reex_env_flag::disabled);
    assert(llama_reex_parse_env_flag("false") == llama_reex_env_flag::disabled);
    assert(llama_reex_parse_env_flag("off") == llama_reex_env_flag::disabled);
    assert(llama_reex_parse_env_flag("maybe") == llama_reex_env_flag::invalid);

    assert(!llama_reex_should_use_custom_unified_split(false, llama_reex_env_flag::enabled));
    assert(!llama_reex_should_use_custom_unified_split(true, llama_reex_env_flag::unset));
    assert(!llama_reex_should_use_custom_unified_split(true, llama_reex_env_flag::disabled));
    assert(!llama_reex_should_use_custom_unified_split(true, llama_reex_env_flag::invalid));
    assert(llama_reex_should_use_custom_unified_split(true, llama_reex_env_flag::enabled));

    return 0;
}
