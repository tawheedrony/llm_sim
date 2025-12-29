#include <stdlib.h>
#include <pthread.h>
#include "kv_backend.h"
#include "sim_config.h"
#include "page_alloc.h"
#include "workload.h"

typedef struct PageSlot {
    Page* page;
} PageSlot;

typedef struct PagedSeqState {
    PageSlot* slots;
    size_t slots_capacity;
    size_t cur_tokens;
    size_t shared_prefix_tokens;
} PagedSeqState;

typedef struct SharedPrefix {
    Page** pages;
    size_t num_pages;
    size_t prefix_tokens;
    int initialized;
} SharedPrefix;

typedef struct PagedKVImpl {
    SimConfig cfg;
    PageAllocator* alloc;

    PagedSeqState* seqs;
    size_t num_seqs;
    size_t seq_capacity;

    SharedPrefix* groups;    // size = cfg.num_groups
    size_t num_groups;

    pthread_mutex_t mutex;
} PagedKVImpl;

static void paged_seq_reserve_slots(PagedSeqState* s, size_t n) {
    if (n <= s->slots_capacity) return;
    size_t new_cap = s->slots_capacity == 0 ? 4 : s->slots_capacity * 2;
    while (new_cap < n) new_cap *= 2;
    PageSlot* ns = (PageSlot*) realloc(s->slots, new_cap * sizeof(PageSlot));
    if (!ns) abort();
    for (size_t i = s->slots_capacity; i < new_cap; ++i) {
        ns[i].page = NULL;
    }
    s->slots = ns;
    s->slots_capacity = new_cap;
}

static SharedPrefix build_shared_prefix(PagedKVImpl* impl, size_t prefix_tokens) {
    SharedPrefix pref = {0};
    if (prefix_tokens == 0) return pref;
    size_t tokens_per_page = impl->cfg.tokens_per_page;
    size_t pages_needed = (prefix_tokens + tokens_per_page - 1) / tokens_per_page;

    pref.pages = (Page**) malloc(pages_needed * sizeof(Page*));
    pref.num_pages = pages_needed;
    pref.prefix_tokens = prefix_tokens;
    pref.initialized = 1;

    for (size_t i = 0; i < pages_needed; ++i) {
        pref.pages[i] = page_alloc(impl->alloc);
    }
    return pref;
}

static size_t shareable_tokens(const PagedKVImpl* impl, size_t tokens) {
    size_t per_page = impl->cfg.tokens_per_page;
    if (per_page == 0) return 0;
    return (tokens / per_page) * per_page;
}

// call after impl is created
static void paged_init_prefix_groups(PagedKVImpl* impl) {
    impl->num_groups = impl->cfg.num_groups;
    if (impl->num_groups == 0) {
        impl->groups = NULL;
        return;
    }
    impl->groups = (SharedPrefix*) calloc(impl->num_groups, sizeof(SharedPrefix));
}

static SeqId paged_init_sequence(KVBackend* backend, const SequenceWork* work) {
    PagedKVImpl* impl = (PagedKVImpl*) backend->impl;
    pthread_mutex_lock(&impl->mutex);

    if (impl->num_seqs == impl->seq_capacity) {
        size_t new_cap = impl->seq_capacity == 0 ? 16 : impl->seq_capacity * 2;
        PagedSeqState* ns = (PagedSeqState*) realloc(impl->seqs, new_cap * sizeof(PagedSeqState));
        if (!ns) {
            pthread_mutex_unlock(&impl->mutex);
            abort();
        }
        for (size_t i = impl->seq_capacity; i < new_cap; ++i) {
            ns[i].slots = NULL;
            ns[i].slots_capacity = 0;
            ns[i].cur_tokens = 0;
            ns[i].shared_prefix_tokens = 0;
        }
        impl->seqs = ns;
        impl->seq_capacity = new_cap;
    }

    SeqId id = impl->num_seqs++;
    PagedSeqState* s = &impl->seqs[id];
    s->cur_tokens = 0;
    s->shared_prefix_tokens = 0;

    const int shared_id = work->shared_prompt_id;
    size_t shared_tokens = (shared_id >= 0) ? shareable_tokens(impl, work->shared_prompt_tokens) : 0;

    if (shared_tokens > 0 && impl->num_groups > 0) {
        size_t gid = (size_t) shared_id % impl->num_groups;
        SharedPrefix* pref = &impl->groups[gid];
        if (!pref->initialized) {
            *pref = build_shared_prefix(impl, shared_tokens);
        }
        if (pref->prefix_tokens != shared_tokens) {
            shared_tokens = pref->prefix_tokens;
        }
        size_t prefix_pages = pref->num_pages;
        paged_seq_reserve_slots(s, prefix_pages);
        for (size_t i = 0; i < prefix_pages; ++i) {
            Page* p = pref->pages[i];
            page_inc_ref(impl->alloc, p);
            s->slots[i].page = p;
        }
        s->shared_prefix_tokens = shared_tokens;
    }

    pthread_mutex_unlock(&impl->mutex);
    return id;
}

static void paged_append_token(KVBackend* backend, SeqId id) {
    PagedKVImpl* impl = (PagedKVImpl*) backend->impl;
    PagedSeqState* s = &impl->seqs[id];

    if (s->cur_tokens >= impl->cfg.max_context_tokens) {
        return;
    }

    size_t idx = s->cur_tokens;
    size_t tokens_per_page = impl->cfg.tokens_per_page;
    size_t page_idx = idx / tokens_per_page;

    if (page_idx >= s->slots_capacity || s->slots[page_idx].page == NULL) {
        pthread_mutex_lock(&impl->mutex);
        if (page_idx >= s->slots_capacity) {
            paged_seq_reserve_slots(s, page_idx + 1);
        }
        if (s->slots[page_idx].page == NULL) {
            s->slots[page_idx].page = page_alloc(impl->alloc);
        }
        pthread_mutex_unlock(&impl->mutex);
    }

    s->cur_tokens = idx + 1;
}

static void paged_finish_sequence(KVBackend* backend, SeqId id) {
    PagedKVImpl* impl = (PagedKVImpl*) backend->impl;
    if (id >= impl->num_seqs) return;
    PagedSeqState* s = &impl->seqs[id];

    pthread_mutex_lock(&impl->mutex);
    for (size_t i = 0; i < s->slots_capacity; ++i) {
        if (s->slots[i].page) {
            page_dec_ref(impl->alloc, s->slots[i].page);
            s->slots[i].page = NULL;
        }
    }
    s->cur_tokens = 0;
    s->shared_prefix_tokens = 0;
    pthread_mutex_unlock(&impl->mutex);
}

static KVStats paged_stats(KVBackend* backend) {
    PagedKVImpl* impl = (PagedKVImpl*) backend->impl;
    KVStats st = (KVStats){0, 0, 0};

    pthread_mutex_lock(&impl->mutex);
    for (size_t i = 0; i < impl->num_seqs; ++i) {
        st.logical_tokens += impl->seqs[i].cur_tokens;
    }
    pthread_mutex_unlock(&impl->mutex);

    st.logical_bytes = st.logical_tokens * bytes_per_token(&impl->cfg);
    size_t pages_in_use = page_allocator_pages_in_use(impl->alloc);
    st.physical_bytes = pages_in_use * page_allocator_page_bytes(impl->alloc);
    return st;
}

static void paged_destroy(KVBackend* backend) {
    PagedKVImpl* impl = (PagedKVImpl*) backend->impl;

    for (size_t i = 0; i < impl->num_seqs; ++i) {
        paged_finish_sequence(backend, i);
        free(impl->seqs[i].slots);
    }
    free(impl->seqs);

    for (size_t g = 0; g < impl->num_groups; ++g) {
        SharedPrefix* pref = &impl->groups[g];
        if (!pref->initialized) continue;
        for (size_t i = 0; i < pref->num_pages; ++i) {
            page_dec_ref(impl->alloc, pref->pages[i]);
        }
        free(pref->pages);
    }
    free(impl->groups);

    page_allocator_destroy(impl->alloc);
    pthread_mutex_destroy(&impl->mutex);
    free(impl);
    backend->impl = NULL;
}

static const KVBackendVTable PAGED_VTABLE = {
    .init_sequence   = paged_init_sequence,
    .append_token    = paged_append_token,
    .finish_sequence = paged_finish_sequence,
    .stats           = paged_stats,
    .destroy         = paged_destroy
};

KVBackend* create_paged_backend(const SimConfig* cfg) {
    KVBackend* b = (KVBackend*) malloc(sizeof(KVBackend));
    PagedKVImpl* impl = (PagedKVImpl*) calloc(1, sizeof(PagedKVImpl));
    impl->cfg   = *cfg;
    impl->alloc = page_allocator_create(cfg);
    pthread_mutex_init(&impl->mutex, NULL);
    paged_init_prefix_groups(impl);

    b->vtable = &PAGED_VTABLE;
    b->impl   = impl;
    return b;
}
