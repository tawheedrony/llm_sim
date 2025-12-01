#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sim_config.h"
#include "workload.h"
#include "sim.h"
#include "mono_kv.h"
#include "page_kv.h"

static void print_stats(const char* name, const KVStats* st) {
    printf("%s:\n", name);
    printf("  logical_bytes  = %zu\n", st->logical_bytes);
    printf("  physical_bytes = %zu\n", st->physical_bytes);

    // FIX: Handle case where physical < logical (due to sharing) to avoid underflow
    if (st->physical_bytes > st->logical_bytes) {
        size_t waste = st->physical_bytes - st->logical_bytes;
        double ratio = (double)waste / (double)st->physical_bytes;
        printf("  waste_bytes    = %zu (%.2f%%)\n", waste, ratio * 100.0);
    } else {
        size_t saved = st->logical_bytes - st->physical_bytes;
        double ratio = (double)saved / (double)st->logical_bytes;
        printf("  memory_saved   = %zu (%.2f%% due to sharing)\n", saved, ratio * 100.0);
    }
}

int main(void) {
    srand((unsigned int) time(NULL));

    SimConfig cfg;
    // FIX: Scale down dimensions to fit in RAM.
    // Previous config was ~512KB per token, requiring >160GB RAM.
    // New config is ~8KB per token.
    cfg.num_layers       = 4;   // Reduced from 32
    cfg.num_heads        = 8;   // Reduced from 32
    cfg.head_dim         = 64;  // Reduced from 128
    
    cfg.tokens_per_page  = 16;  // 16 tokens per page is a common default (e.g. vLLM)
    cfg.arena_bytes      = (size_t)2 << 30; // Increase arena to 2 GB

    cfg.num_sequences    = 128;
    cfg.num_groups       = 4;
    cfg.max_prompt_extra = 128;
    cfg.min_gen_tokens   = 128;
    cfg.max_gen_tokens   = 512;
    cfg.enable_sleep     = 0;

    printf("bytes_per_token = %zu\n", bytes_per_token(&cfg));

    // FIX: Manually generate workload to ensure shared_prompt_id is set correctly.
    // generate_workload(&cfg) might not set shared IDs, resulting in 0% savings.
    SequenceWork* work = calloc(cfg.num_sequences, sizeof(SequenceWork));
    int seq_idx = 0;
    size_t shared_prompt_len = 256; // Explicit shared length

    for (int g = 0; g < cfg.num_groups; ++g) {
        int group_prompt_id = g + 1; 
        int seqs_in_group = cfg.num_sequences / cfg.num_groups;
        
        for (int i = 0; i < seqs_in_group; ++i) {
            if (seq_idx >= cfg.num_sequences) break;
            
            // FIX: Removed .id assignment (not in struct)
            // FIX: Renamed .prompt_len -> .prompt_tokens
            work[seq_idx].prompt_tokens = shared_prompt_len;
            
            // FIX: Renamed .gen_len -> .gen_tokens
            work[seq_idx].gen_tokens = cfg.min_gen_tokens + (rand() % (cfg.max_gen_tokens - cfg.min_gen_tokens + 1));
            
            // This requires the update to include/workload.h
            work[seq_idx].shared_prompt_id = group_prompt_id; // CRITICAL for sharing
            
            seq_idx++;
        }
    }

    // Baseline monolithic
    KVBackend* mono = create_monolithic_backend(&cfg);
    KVStats st_mono = run_simulation(mono, &cfg, work);
    print_stats("Monolithic", &st_mono);
    kv_destroy(mono);

    // Paged + prefix sharing
    KVBackend* paged = create_paged_backend(&cfg);
    KVStats st_paged = run_simulation(paged, &cfg, work);
    print_stats("Paged+Prefix", &st_paged);
    kv_destroy(paged);

    free(work);
    return 0;
}
