#ifndef WORKLOAD_H
#define WORKLOAD_H

#include <stddef.h>
#include "sim_config.h"

typedef struct {
    size_t prompt_tokens;        // includes any shared prefix
    size_t gen_tokens;
    size_t shared_prompt_tokens; // shareable prefix (must be page-aligned)
    int    shared_prompt_id;     // -1 => no sharing
} SequenceWork;

// Generate an array of SequenceWork of length num_sequences
// Caller owns returned pointer; free() when done.
SequenceWork* generate_workload(const SimConfig* cfg);

#endif
