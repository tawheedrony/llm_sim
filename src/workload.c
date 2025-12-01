#include <stdlib.h>
#include "sim_config.h"
#include "workload.h"

SequenceWork* generate_workload(const SimConfig* cfg) {
    SequenceWork* w = (SequenceWork*) malloc(cfg->num_sequences * sizeof(SequenceWork));
    if (!w) abort();

    size_t tokens_per_page = cfg->tokens_per_page ? cfg->tokens_per_page : 1;
    size_t base_prefix = tokens_per_page * 128;
    size_t shareable_prefix = (base_prefix / tokens_per_page) * tokens_per_page;

    for (size_t i = 0; i < cfg->num_sequences; ++i) {
        int group = cfg->num_groups ? (int)(i % cfg->num_groups) : -1;
        w[i].shared_prompt_id = group;
        w[i].shared_prompt_tokens = (group >= 0) ? shareable_prefix : 0;

        size_t extra_prompt = rand() % (cfg->max_prompt_extra + 1);
        size_t prompt_base = (group >= 0) ? w[i].shared_prompt_tokens : 0;
        w[i].prompt_tokens = prompt_base + extra_prompt;

        size_t gen_span = cfg->max_gen_tokens - cfg->min_gen_tokens + 1;
        w[i].gen_tokens = cfg->min_gen_tokens + (rand() % gen_span);
    }
    return w;
}
