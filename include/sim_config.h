#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

#include <stddef.h>
#include <stdint.h>

typedef struct SimConfig {
    size_t num_layers;
    size_t num_heads;
    size_t head_dim;
    size_t tokens_per_page;
    size_t arena_bytes;

    size_t num_sequences;
    size_t num_groups;         // how many shared-prefix groups
    size_t max_prompt_extra;   // extra tokens on top of prefix
    size_t min_gen_tokens;
    size_t max_gen_tokens;

    int    enable_sleep;       // non-zero: simulate compute with usleep
} SimConfig;

static inline size_t bytes_per_token(const SimConfig* cfg) {
    // 2 for K and V, assume fp16 (2 bytes)
    return cfg->num_layers * cfg->num_heads * cfg->head_dim * 2u * 2u;
}

#endif
