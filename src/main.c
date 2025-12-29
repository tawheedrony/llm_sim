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
    cfg.num_layers       = 4;
    cfg.num_heads        = 8;
    cfg.head_dim         = 64;

    cfg.max_context_tokens = 2048;     // NEW: realistic window

    cfg.tokens_per_page  = 16;         // common-ish simulator choice
    cfg.arena_bytes      = (size_t)2 << 30; // 2 GiB arena

    cfg.num_sequences    = 128;
    cfg.num_groups       = 4;          // enables prefix sharing groups
    cfg.max_prompt_extra = 256;
    cfg.min_gen_tokens   = 128;
    cfg.max_gen_tokens   = 1024;
    cfg.enable_sleep     = 0;

    printf("bytes_per_token = %zu\n", bytes_per_token(&cfg));

    SequenceWork* work = generate_workload(&cfg);

    KVBackend* mono = create_monolithic_backend(&cfg);
    KVStats st_mono = run_simulation(mono, &cfg, work);
    print_stats("Monolithic (fixed 2048)", &st_mono);
    kv_destroy(mono);

    KVBackend* paged = create_paged_backend(&cfg);
    KVStats st_paged = run_simulation(paged, &cfg, work);
    print_stats("Paged+Prefix (max 2048)", &st_paged);
    kv_destroy(paged);

    free(work);
    return 0;
}
