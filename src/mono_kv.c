#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "kv_backend.h"
#include "sim_config.h"

typedef struct MonoSeqState {
    size_t max_tokens;
    size_t cur_tokens;
    size_t bytes_per_token;
    unsigned char* kv_buffer; // optional: to stress RSS
} MonoSeqState;

typedef struct MonoKVImpl {
    SimConfig cfg;
    MonoSeqState* seqs;
    size_t num_seqs;
    size_t capacity;
    pthread_mutex_t mutex;
} MonoKVImpl;

static SeqId mono_init_sequence(KVBackend* backend, const SequenceWork* work) {
    MonoKVImpl* impl = (MonoKVImpl*) backend->impl;
    pthread_mutex_lock(&impl->mutex);

    if (impl->num_seqs == impl->capacity) {
        size_t new_cap = impl->capacity == 0 ? 16 : impl->capacity * 2;
        MonoSeqState* ns = (MonoSeqState*) realloc(impl->seqs, new_cap * sizeof(MonoSeqState));
        if (!ns) {
            pthread_mutex_unlock(&impl->mutex);
            abort();
        }
        impl->seqs = ns;
        impl->capacity = new_cap;
    }

    SeqId id = impl->num_seqs++;
    MonoSeqState* s = &impl->seqs[id];
    s->bytes_per_token = bytes_per_token(&impl->cfg);
    
    // FIX: Allocate a fixed "Context Window" size to simulate real-world monolithic waste.
    // Real systems must pre-allocate max_context_length because they can't realloc easily.
    s->max_tokens = 4096; 
    
    s->cur_tokens = 0;
    s->kv_buffer = (unsigned char*) malloc(s->max_tokens * s->bytes_per_token);
    if (!s->kv_buffer) {
        pthread_mutex_unlock(&impl->mutex);
        abort();
    }

    pthread_mutex_unlock(&impl->mutex);
    return id;
}

static void mono_append_token(KVBackend* backend, SeqId id) {
    MonoKVImpl* impl = (MonoKVImpl*) backend->impl;
    MonoSeqState* s = &impl->seqs[id];
    if (s->cur_tokens < s->max_tokens) {
        s->cur_tokens++;
    }
}

static void mono_finish_sequence(KVBackend* backend, SeqId id) {
    (void) backend;
    (void) id;
    // no-op; keep them until end for stats
}

static KVStats mono_stats(KVBackend* backend) {
    MonoKVImpl* impl = (MonoKVImpl*) backend->impl;
    KVStats st = {0, 0, 0};

    pthread_mutex_lock(&impl->mutex);
    for (size_t i = 0; i < impl->num_seqs; ++i) {
        MonoSeqState* s = &impl->seqs[i];
        st.logical_tokens += s->cur_tokens;
        st.physical_bytes += s->max_tokens * s->bytes_per_token;
    }
    pthread_mutex_unlock(&impl->mutex);

    st.logical_bytes = st.logical_tokens * bytes_per_token(&impl->cfg);
    return st;
}

static void mono_destroy(KVBackend* backend) {
    MonoKVImpl* impl = (MonoKVImpl*) backend->impl;
    for (size_t i = 0; i < impl->num_seqs; ++i) {
        free(impl->seqs[i].kv_buffer);
    }
    free(impl->seqs);
    pthread_mutex_destroy(&impl->mutex);
    free(impl);
    backend->impl = NULL;
}

// Ensure this VTable is defined (it was likely already there based on the warning)
static const KVBackendVTable MONO_VTABLE = {
    .init_sequence   = mono_init_sequence,
    .append_token    = mono_append_token,
    .finish_sequence = mono_finish_sequence,
    .stats           = mono_stats,
    .destroy         = mono_destroy
};

KVBackend* create_monolithic_backend(const SimConfig* cfg) {
    MonoKVImpl* impl = (MonoKVImpl*) calloc(1, sizeof(MonoKVImpl));
    impl->cfg = *cfg;
    pthread_mutex_init(&impl->mutex, NULL);
    
    // Pre-allocate capacity
    impl->capacity = cfg->num_sequences;
    impl->seqs = (MonoSeqState*) calloc(impl->capacity, sizeof(MonoSeqState));
    
    KVBackend* b = (KVBackend*) calloc(1, sizeof(KVBackend));
    b->impl = impl;
    
    // FIX: Use the vtable instead of direct assignment
    b->vtable = &MONO_VTABLE;
    
    return b;
}
