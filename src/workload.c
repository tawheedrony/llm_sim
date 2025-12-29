#include <stdlib.h>
#include "sim_config.h"
#include "workload.h"

static size_t align_down(size_t x, size_t a) {
    if (a == 0) return 0;
    return (x / a) * a;
}

SequenceWork* generate_workload(const SimConfig* cfg) {
    SequenceWork* w = (SequenceWork*) malloc(cfg->num_sequences * sizeof(SequenceWork));
    if (!w) abort();

    const size_t tpp = cfg->tokens_per_page ? cfg->tokens_per_page : 1;
    const size_t max_ctx = cfg->max_context_tokens ? cfg->max_context_tokens : 2048;

    // Make prefix substantial but not the whole window (realistic sharing)
    // e.g., 1024 tokens if max_ctx=2048 and tpp=16.
    size_t target_prefix = max_ctx / 2;
    size_t shareable_prefix = align_down(target_prefix, tpp);

    for (size_t i = 0; i < cfg->num_sequences; ++i) {
        int group = cfg->num_groups ? (int)(i % cfg->num_groups) : -1;
        w[i].shared_prompt_id = group;

        w[i].shared_prompt_tokens = (group >= 0) ? shareable_prefix : 0;

        // Prompt = shared_prefix + extra (but <= max_ctx)
        size_t extra_prompt = (cfg->max_prompt_extra > 0)
            ? (size_t)(rand() % (cfg->max_prompt_extra + 1))
            : 0;

        size_t prompt = w[i].shared_prompt_tokens + extra_prompt;
        if (prompt > max_ctx) prompt = max_ctx;
        w[i].prompt_tokens = prompt;

        // Gen tokens sampled but clipped so prompt+gen <= max_ctx
        size_t remaining = (prompt < max_ctx) ? (max_ctx - prompt) : 0;
        size_t gen_min = cfg->min_gen_tokens;
        size_t gen_max = cfg->max_gen_tokens;

        if (gen_min > gen_max) gen_min = gen_max;

        size_t gen_span = (gen_max >= gen_min) ? (gen_max - gen_min + 1) : 1;
        size_t gen = gen_min + (size_t)(rand() % gen_span);

        if (gen > remaining) gen = remaining;
        w[i].gen_tokens = gen;
    }
    return w;
}
