#ifndef KV_BACKEND_H
#define KV_BACKEND_H

#include <stddef.h>
#include <stdlib.h>
#include "sim_config.h"
#include "workload.h"

typedef size_t SeqId;

typedef struct KVStats {
    size_t logical_tokens;
    size_t logical_bytes;
    size_t physical_bytes;
} KVStats;

struct KVBackend;

typedef struct KVBackendVTable {
    SeqId  (*init_sequence)(struct KVBackend* backend, const SequenceWork* work);
    void   (*append_token)(struct KVBackend* backend, SeqId id);
    void   (*finish_sequence)(struct KVBackend* backend, SeqId id);
    KVStats (*stats)(struct KVBackend* backend);
    void   (*destroy)(struct KVBackend* backend);
} KVBackendVTable;

typedef struct KVBackend {
    const KVBackendVTable* vtable;
    void* impl;
} KVBackend;

// Helper inline wrappers
static inline SeqId kv_init_sequence(KVBackend* b, const SequenceWork* w) {
    return b->vtable->init_sequence(b, w);
}
static inline void kv_append_token(KVBackend* b, SeqId id) {
    b->vtable->append_token(b, id);
}
static inline void kv_finish_sequence(KVBackend* b, SeqId id) {
    b->vtable->finish_sequence(b, id);
}
static inline KVStats kv_stats(KVBackend* b) {
    return b->vtable->stats(b);
}
static inline void kv_destroy(KVBackend* b) {
    if (!b) return;
    b->vtable->destroy(b);
    free(b);
}

#endif
