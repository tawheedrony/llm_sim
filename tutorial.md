This repository is a small C simulator that models memory usage of large language model (LLM) KV caches under concurrent decoding. It compares:

- A **monolithic** KV cache: each sequence gets a fixed, preallocated context window.
- A **paged** KV cache with **shared prefixes**: sequences share page-aligned prefixes and allocate pages on demand.

The goal is to show that paging + prefix sharing drastically reduce memory waste.

Below is a detailed walkthrough file by file and then concept by concept.

---

## Core concepts

### SimConfig (sim_config.h)

The configuration struct:

```c
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
```

It captures:

- **Model shape**: $L = \text{num_layers}$, $H = \text{num_heads}$, $D = \text{head_dim}$.
- **Paging parameters**:
  - `tokens_per_page`: how many tokens per page.
  - `arena_bytes`: how big the backing “GPU memory” arena is (for the paged backend).
- **Workload parameters**:
  - `num_sequences`: how many concurrent decoding sequences.
  - `num_groups`: how many shared-prompt groups.
  - `max_prompt_extra`: extra random prompt tokens on top of shared prefix.
  - `min_gen_tokens`, `max_gen_tokens`: generation length range per sequence.
- `enable_sleep`: if non-zero, `usleep` is used to simulate compute time per token.

The KV size per token:

```c
static inline size_t bytes_per_token(const SimConfig* cfg) {
    // 2 for K and V, assume fp16 (2 bytes)
    return cfg->num_layers * cfg->num_heads * cfg->head_dim * 2u * 2u;
}
```

This models KV cache cost as:

$$
\text{bytes\_per\_token} = L \cdot H \cdot D \cdot 2 \cdot 2
$$

where:
- factor $2$ for $K$ vs $V$,
- factor $2$ for 2 bytes per fp16 value.

---

### Workload representation (workload.h, workload.c)

A **sequence** in the workload:

```c
typedef struct {
    size_t prompt_tokens;        // includes any shared prefix
    size_t gen_tokens;
    size_t shared_prompt_tokens; // shareable prefix (must be page-aligned)
    int    shared_prompt_id;     // -1 => no sharing
} SequenceWork;
```

Fields:

- `prompt_tokens`: total prompt length including any shared part.
- `gen_tokens`: how many tokens to generate.
- `shared_prompt_tokens`: how many of those prompt tokens come from a shareable prefix. The comment says it “must be page-aligned” for clean sharing.
- `shared_prompt_id`:
  - `-1` means no sharing.
  - `>= 0` groups sequences that share a prefix; sequences with same `shared_prompt_id` can share pages.

#### Automatic workload generator (workload.c)

```c
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
```

Key behaviors:

- Uses `cfg->num_groups` to cyclically assign sequences to groups (`i % num_groups`) or `-1` if no groups.
- `shared_prompt_tokens`:
  - uses `base_prefix = tokens_per_page * 128`, then `shareable_prefix` ensures it’s a multiple of `tokens_per_page`, i.e. page-aligned.
- Each sequence:
  - `shared_prompt_id`: group or `-1`.
  - `shared_prompt_tokens`: page-aligned prefix if grouped.
  - `prompt_tokens = shared_prefix + extra_prompt`, where:
    - `extra_prompt` is uniform in $[0, \text{max\_prompt\_extra}]$.
  - `gen_tokens` is uniform in $[\text{min\_gen\_tokens}, \text{max\_gen\_tokens}]$.

**Note:** In main.c, the author currently *does not* call `generate_workload`; they manually construct `SequenceWork` instead for more controlled sharing.

---

### Backend abstraction (kv_backend.h)

Defines the generic KV backend interface.

```c
typedef size_t SeqId;

typedef struct KVStats {
    size_t logical_tokens;
    size_t logical_bytes;
    size_t physical_bytes;
} KVStats;
```

Semantics:

- `logical_tokens`: sum of tokens across all sequences (conceptual tokens).
- `logical_bytes`: `logical_tokens * bytes_per_token(cfg)`; how much memory you’d need if no sharing / no fragmentation.
- `physical_bytes`: actual bytes reserved/allocated by the backend.

Virtual table:

```c
typedef struct KVBackendVTable {
    SeqId  (*init_sequence)(struct KVBackend* backend, const SequenceWork* work);
    void   (*append_token)(struct KVBackend* backend, SeqId id);
    void   (*finish_sequence)(struct KVBackend* backend, SeqId id);
    KVStats (*stats)(struct KVBackend* backend);
    void   (*destroy)(struct KVBackend* backend);
} KVBackendVTable;

typedef struct KVBackend {
    const KVBackendVTable* vtable;
    void* impl; // backend-specific state
} KVBackend;
```

Inline wrappers for convenience:

```c
static inline SeqId kv_init_sequence(KVBackend* b, const SequenceWork* w) {
    return b->vtable->init_sequence(b, w);
}
// etc.
```

This allows multiple implementations—`mono_kv` and `page_kv`—to share the same driver code (sim.c) via this abstraction.

---

### Simulation driver (sim.h, sim.c)

sim.h:

```c
KVStats run_simulation(KVBackend* backend,
                       const SimConfig* cfg,
                       const SequenceWork* work);
```

sim.c:

- Uses POSIX threads to simulate concurrent decoding across sequences.
- Each thread processes one sequence from the given workload.

#### Thread function

```c
typedef struct ThreadArgs {
    KVBackend* backend;
    const SimConfig* cfg;
    const SequenceWork* work;
    size_t index;
} ThreadArgs;
```

The per-thread worker:

```c
static void* decode_thread(void* arg) {
    ThreadArgs* a = (ThreadArgs*) arg;
    const SequenceWork* w = &a->work[a->index];

    SeqId id = kv_init_sequence(a->backend, w);

    // Prompt
    for (size_t t = 0; t < w->prompt_tokens; ++t) {
        kv_append_token(a->backend, id);
        if (a->cfg->enable_sleep) {
            usleep(100); // 0.1 ms
        }
    }
    // Decode
    for (size_t t = 0; t < w->gen_tokens; ++t) {
        kv_append_token(a->backend, id);
        if (a->cfg->enable_sleep) {
            usleep(100);
        }
    }

    // NOTE: They intentionally do not call kv_finish_sequence().
    return NULL;
}
```

Observations:

- `kv_init_sequence` is given a pointer to this sequence’s `SequenceWork`, so backends can use group info and shared prefix metadata to decide memory layout.
- Both prompt and generation loops just call `kv_append_token` once per token.
- `kv_finish_sequence` is intentionally commented out:

  > Do NOT finish the sequence here. We want to measure memory usage while all sequences are active (Peak Memory). Cleanup will happen in kv_destroy().

This means memory is reported at a near-peak: every sequence has completed its prompt + generation, and none has been freed yet.

#### Running the simulation

```c
KVStats run_simulation(KVBackend* backend,
                       const SimConfig* cfg,
                       const SequenceWork* work) {
    size_t n = cfg->num_sequences;
    pthread_t* threads = (pthread_t*) malloc(n * sizeof(pthread_t));
    ThreadArgs* args   = (ThreadArgs*) malloc(n * sizeof(ThreadArgs));

    for (size_t i = 0; i < n; ++i) {
        args[i].backend = backend;
        args[i].cfg     = cfg;
        args[i].work    = work;
        args[i].index   = i;
        pthread_create(&threads[i], NULL, decode_thread, &args[i]);
    }

    for (size_t i = 0; i < n; ++i) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(args);

    return kv_stats(backend);
}
```

- Spawns one thread per sequence.
- Each thread runs `decode_thread`.
- Waits for all threads to finish, then calls backend’s `stats` and returns it.

---

## Monolithic backend (mono_kv.h, mono_kv.c)

This backend models a simple, wasteful KV cache where each sequence pre-reserves a large contiguous buffer (a fixed context window), no sharing.

### Internal state

```c
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
```

- `MonoSeqState`:
  - `max_tokens`: capacity of this sequence’s buffer (fixed context window).
  - `cur_tokens`: how many tokens have been appended so far.
  - `bytes_per_token`: cached from `bytes_per_token(&cfg)`.
  - `kv_buffer`: actual backing memory (just to consume RSS; contents are irrelevant).
- `MonoKVImpl`:
  - `seqs`: array of sequence states, allocated up to `capacity`.
  - `num_seqs`: how many sequences have been initialized.
  - `mutex`: to protect concurrent initialization & stats.

### Sequence initialization

```c
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
    
    // Fixed "context window" for all sequences
    s->max_tokens = 2048; 
    
    s->cur_tokens = 0;
    s->kv_buffer = (unsigned char*) malloc(s->max_tokens * s->bytes_per_token);
    if (!s->kv_buffer) {
        pthread_mutex_unlock(&impl->mutex);
        abort();
    }

    pthread_mutex_unlock(&impl->mutex);
    return id;
}
```

Notable properties:

- Ignores `work` details for capacity: always allocates `max_tokens = 2048` per sequence.
- Allocates `kv_buffer` of size $2048 \cdot \text{bytes\_per\_token}$.
- The 2048 context window is fixed; even if a sequence uses fewer tokens, those bytes are still reserved → *waste*.

### Appending tokens

```c
static void mono_append_token(KVBackend* backend, SeqId id) {
    MonoKVImpl* impl = (MonoKVImpl*) backend->impl;
    MonoSeqState* s = &impl->seqs[id];
    if (s->cur_tokens < s->max_tokens) {
        s->cur_tokens++;
    }
}
```

- Just increments `cur_tokens` until it reaches `max_tokens`; simulation does not enforce overflow beyond that.

### Finishing sequences

```c
static void mono_finish_sequence(KVBackend* backend, SeqId id) {
    (void) backend;
    (void) id;
    // no-op; keep them until end for stats
}
```

- Intentional no-op during the simulation: we don’t free per-sequence buffers until `destroy`.

### Stats

```c
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
```

- `logical_tokens`: sum of `cur_tokens`.
- `physical_bytes`: **sum of full buffer sizes** for all sequences, not just used tokens.

If many sequences use fewer than `max_tokens`, you get large waste:

$$
\text{waste\_bytes} = \text{physical\_bytes} - \text{logical\_bytes} \ge 0
$$

since every sequence gets capacity 2048 tokens, but may use much less.

### Destroy

```c
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
```

### Backend creation

```c
KVBackend* create_monolithic_backend(const SimConfig* cfg) {
    MonoKVImpl* impl = (MonoKVImpl*) calloc(1, sizeof(MonoKVImpl));
    impl->cfg = *cfg;
    pthread_mutex_init(&impl->mutex, NULL);
    
    // Pre-allocate capacity
    impl->capacity = cfg->num_sequences;
    impl->seqs = (MonoSeqState*) calloc(impl->capacity, sizeof(MonoSeqState));
    
    KVBackend* b = (KVBackend*) calloc(1, sizeof(KVBackend));
    b->impl = impl;
    
    b->vtable = &MONO_VTABLE;
    
    return b;
}
```

- Pre-allocates `seqs` array sized to `num_sequences` so it doesn’t need to grow much during the run.
- No memory is allocated yet for `kv_buffer`; that happens per-sequence in `mono_init_sequence`.

---

## Paged backend (page_alloc.h, page_alloc.c, page_kv.h, page_kv.c)

This backend models a realistic paging KV cache with:

- A big arena of bytes, sliced into fixed-size pages.
- Reference counting for each page.
- Multiple sequences that point to pages; they can share pages for the common prefix.

### Page allocator (page_alloc.c)

`Page` and `PageAllocator` are internal (declared opaque in the header):

```c
typedef struct Page {
    unsigned char* base;
    unsigned int ref;
} Page;

typedef struct PageAllocator {
    unsigned char* arena;
    size_t page_bytes;
    size_t num_pages;
    Page*  pages;

    Page** free_list;
    size_t free_count;
    size_t free_capacity;

    pthread_mutex_t mutex;
} PageAllocator;
```

Key responsibilities:

- Manage a big continuous arena via `mmap`.
- Slice the arena into `num_pages` pages of size `page_bytes`.
- Track free pages via `free_list`.
- Maintain `ref` counts and return pages to the free list when `ref` hits 0.

#### Creation

```c
PageAllocator* page_allocator_create(const SimConfig* cfg) {
    PageAllocator* pa = (PageAllocator*) calloc(1, sizeof(PageAllocator));
    if (!pa) abort();

    pa->page_bytes = cfg->tokens_per_page * bytes_per_token(cfg);
    pa->num_pages  = cfg->arena_bytes / pa->page_bytes;

    size_t arena_size = pa->num_pages * pa->page_bytes;
    pa->arena = (unsigned char*) mmap(NULL, arena_size,
                                      PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ...
    pa->pages = (Page*) malloc(pa->num_pages * sizeof(Page));
    pa->free_list = (Page**) malloc(pa->num_pages * sizeof(Page*));
    pa->free_capacity = pa->num_pages;
    pa->free_count = 0;

    for (size_t i = 0; i < pa->num_pages; ++i) {
        pa->pages[i].base = pa->arena + i * pa->page_bytes;
        pa->pages[i].ref  = 0;
        pa->free_list[pa->free_count++] = &pa->pages[i];
    }

    pthread_mutex_init(&pa->mutex, NULL);
    return pa;
}
```

- `page_bytes = tokens_per_page * bytes_per_token(cfg)`.
- `num_pages = arena_bytes / page_bytes`.
- `arena` is an `mmap`-allocated region.
- Fills `pages[i].base` to point within the arena.
- All pages start with `ref=0`, in the `free_list`.

#### Allocation and reference counting

Allocate a page:

```c
Page* page_alloc(PageAllocator* pa) {
    pthread_mutex_lock(&pa->mutex);
    if (pa->free_count == 0) {
        pthread_mutex_unlock(&pa->mutex);
        abort(); // out of pages
    }
    Page* p = pa->free_list[--pa->free_count];
    p->ref = 1;
    pthread_mutex_unlock(&pa->mutex);
    return p;
}
```

Increase ref:

```c
void page_inc_ref(PageAllocator* pa, Page* p) {
    (void) pa;
    p->ref++;
}
```

- Non-atomic increment is OK for the simulator.

Decrease ref:

```c
void page_dec_ref(PageAllocator* pa, Page* p) {
    pthread_mutex_lock(&pa->mutex);
    if (p->ref == 0) {
        pthread_mutex_unlock(&pa->mutex);
        abort();
    }
    p->ref--;
    if (p->ref == 0) {
        pa->free_list[pa->free_count++] = p;
    }
    pthread_mutex_unlock(&pa->mutex);
}
```

- When `ref` hits 0, the page is returned to `free_list`.

Page usage stats:

```c
size_t page_allocator_pages_in_use(PageAllocator* pa) {
    size_t used = 0;
    pthread_mutex_lock(&pa->mutex);
    for (size_t i = 0; i < pa->num_pages; ++i) {
        if (pa->pages[i].ref > 0) used++;
    }
    pthread_mutex_unlock(&pa->mutex);
    return used;
}

size_t page_allocator_page_bytes(PageAllocator* pa) {
    return pa->page_bytes;
}
```

Destroy:

```c
void page_allocator_destroy(PageAllocator* pa) {
    size_t arena_size = pa->num_pages * pa->page_bytes;
    munmap(pa->arena, arena_size);
    free(pa->pages);
    free(pa->free_list);
    pthread_mutex_destroy(&pa->mutex);
    free(pa);
}
```

---

### Paged KV backend (page_kv.c)

Public interface in page_kv.h:

```c
KVBackend* create_paged_backend(const SimConfig* cfg);
```

#### Data structures

Sequence representation in paged backend:

```c
typedef struct PageSlot {
    Page* page;
} PageSlot;

typedef struct PagedSeqState {
    PageSlot* slots;
    size_t slots_capacity;
    size_t cur_tokens;
    size_t shared_prefix_tokens;
} PagedSeqState;
```

- Each sequence is represented by a vector of `PageSlot`s:
  - `slots[i].page` is the page used for tokens `[i * tokens_per_page, (i+1)*tokens_per_page)`.
- `cur_tokens`: how many tokens have been appended.
- `shared_prefix_tokens`: how many prefix tokens are shared (page-aligned).

Shared-prefix representation (per group):

```c
typedef struct SharedPrefix {
    Page** pages;
    size_t num_pages;
    size_t prefix_tokens;
    int initialized;
} SharedPrefix;
```

- A prefix is defined by:
  - An array of page pointers.
  - `prefix_tokens`: number of tokens this prefix covers.
- `initialized` indicates if pages have already been allocated for this group.

Backend implementation:

```c
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
```

- `cfg`: configuration copy.
- `alloc`: pointer to the shared `PageAllocator` described earlier.
- `seqs`: array of `PagedSeqState`.
- `groups`: array of `SharedPrefix` for each group.
- `mutex`: protects sequence allocation, shared group setup, and destruction.

#### Helper: reserve slots for a sequence

```c
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
```

- Grows `slots` capacity geometrically.
- Initializes new entries with `page = NULL`.

#### Helper: building shared prefix

```c
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
```

- Allocates enough pages to cover `prefix_tokens`, rounding up to pages.

Helper: compute shareable tokens:

```c
static size_t shareable_tokens(const PagedKVImpl* impl, size_t tokens) {
    size_t per_page = impl->cfg.tokens_per_page;
    if (per_page == 0) return 0;
    return (tokens / per_page) * per_page;
}
```

- Ensures shared tokens are a multiple of `tokens_per_page` (page-aligned).

Initialize groups array:

```c
static void paged_init_prefix_groups(PagedKVImpl* impl) {
    impl->num_groups = impl->cfg.num_groups;
    if (impl->num_groups == 0) {
        impl->groups = NULL;
        return;
    }
    impl->groups = (SharedPrefix*) calloc(impl->num_groups, sizeof(SharedPrefix));
}
```

#### Sequence initialization

```c
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
```

Logic:

- Expand `seqs` array if needed, initializing new entries.
- Compute how many `shared_tokens`:
  - Only if `shared_prompt_id >= 0` and there is at least one group.
  - `shareable_tokens` enforces page alignment.
- If sharing is possible:
  - Determine group index `gid`.
  - If this group’s shared prefix is not initialized:
    - Build it (allocate pages from `PageAllocator`).
  - If the existing prefix has a different `prefix_tokens`, clamp `shared_tokens` to the existing prefix.
  - Reserve enough slots for these prefix pages.
  - For each prefix page:
    - `page_inc_ref` (since we now share that page).
    - Assign `s->slots[i].page` = that page.
  - Store `shared_prefix_tokens`.

Thus all sequences with the same `gid` share the exact same `Page*` pointers for their initial pages, and only reference counts change.

#### Appending tokens

```c
static void paged_append_token(KVBackend* backend, SeqId id) {
    PagedKVImpl* impl = (PagedKVImpl*) backend->impl;
    PagedSeqState* s = &impl->seqs[id];

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
```

Mechanics:

- Compute which page index the new token belongs to:
  - $ \text{page\_idx} = \lfloor \text{cur\_tokens} / \text{tokens\_per\_page} \rfloor $.
- If that page slot is missing or has no page yet:
  - Lock `mutex`.
  - Grow slots if necessary.
  - Allocate a new page via `page_alloc` if `page == NULL`.
  - Unlock.
- Then increment `cur_tokens`.

This means:

- Pages are allocated **on demand** only when tokens actually arrive.
- Shared prefix pages were already attached in `init_sequence`, so early tokens reuse those.

#### Finish sequence

```c
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
```

- Decrements refcount for each page slot:
  - When `ref` hits 0, the page returns to `free_list` in `PageAllocator`.
- Resets `cur_tokens` and `shared_prefix_tokens`.

In the current simulation, `finish_sequence` is only called indirectly from `paged_destroy`, not from sim.c.

#### Stats

```c
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
```

- `logical_tokens`: sum over all sequences.
- `logical_bytes`: same formula as monolithic.
- `physical_bytes`: derived from page allocator’s currently used pages:

  $$
  \text{physical\_bytes} = \text{pages\_in\_use} \cdot \text{page\_bytes}
  $$

Here is where prefix sharing and on-demand paging produce big savings:
- If many sequences share the same prefix pages, `pages_in_use` stays relatively low.

#### Destroy

```c
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
```

- Calls `paged_finish_sequence` on all sequences, then frees `slots`.
- Then, for each group’s shared prefix:
  - Decrements ref count on all prefix pages.
- Finally destroys the page allocator and frees the backend impl.

#### Create backend

```c
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
```

---

## Main program (`src/main.c`)

`main` configures a simulation, constructs a workload, runs both backends, and prints memory statistics.

### Printing stats

```c
static void print_stats(const char* name, const KVStats* st) {
    printf("%s:\n", name);
    printf("  logical_bytes  = %zu\n", st->logical_bytes);
    printf("  physical_bytes = %zu\n", st->physical_bytes);

    // Handle case where physical < logical (due to sharing)
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
```

- If `physical_bytes >= logical_bytes`, it reports “waste”.
- If `physical_bytes < logical_bytes`, it reports “memory_saved” due to sharing.
- This is where you see that the paged shared-prefix backend uses less memory than the theoretical no-sharing baseline.

### Config setup

```c
int main(void) {
    srand((unsigned int) time(NULL));

    SimConfig cfg;
    // Scale-down config so it fits RAM, but still large enough to be realistic.
    cfg.num_layers       = 4;   // Reduced from 32
    cfg.num_heads        = 8;   // Reduced from 32
    cfg.head_dim         = 64;  // Reduced from 128
    
    cfg.tokens_per_page  = 16;  
    cfg.arena_bytes      = (size_t)2 << 30; // 2 GB arena

    cfg.num_sequences    = 128;
    cfg.num_groups       = 4;
    cfg.max_prompt_extra = 128;
    cfg.min_gen_tokens   = 128;
    cfg.max_gen_tokens   = 512;
    cfg.enable_sleep     = 0;

    printf("bytes_per_token = %zu\n", bytes_per_token(&cfg));
```

So:

- `bytes_per_token = 4 * 8 * 64 * 2 * 2 = 8192` bytes (8 KB per token).
- Each page: `tokens_per_page = 16` → $16 \cdot 8192 = 131072$ bytes (128 KB per page).
- `arena_bytes = 2 GB`, so approximately $2 \text{GB} / 128\text{KB} \approx 16384$ pages.

### Manual workload construction

Even though `generate_workload` exists, the main uses a custom workload to guarantee sharing:

```c
    // Manually generate workload to ensure shared_prompt_id is set correctly.
    SequenceWork* work = calloc(cfg.num_sequences, sizeof(SequenceWork));
    int seq_idx = 0;
    size_t shared_prompt_len = 256; // Explicit shared length

    for (int g = 0; g < cfg.num_groups; ++g) {
        int group_prompt_id = g + 1; 
        int seqs_in_group = cfg.num_sequences / cfg.num_groups;
        
        for (int i = 0; i < seqs_in_group; ++i) {
            if (seq_idx >= cfg.num_sequences) break;
            
            work[seq_idx].prompt_tokens = shared_prompt_len;
            work[seq_idx].gen_tokens = cfg.min_gen_tokens + (rand() % (cfg.max_gen_tokens - cfg.min_gen_tokens + 1));
            work[seq_idx].shared_prompt_id = group_prompt_id; // CRITICAL for sharing
            
            seq_idx++;
        }
    }
```

Notes:

- `shared_prompt_len = 256` tokens.
- For each group `g`:
  - `group_prompt_id = g + 1` (so IDs are 1..num_groups).
  - `seqs_in_group = num_sequences / num_groups = 128 / 4 = 32`.
- Each sequence in a group:
  - `prompt_tokens` = 256 (all of it is effectively shared prefix in concept).
  - `gen_tokens` assigned randomly in the config range.
  - `shared_prompt_id` set to `group_prompt_id`.

**Important subtlety:** they do **not** populate `shared_prompt_tokens` here, only `prompt_tokens` and `shared_prompt_id`. However, in the paged backend’s `paged_init_sequence`, the shared length is read from `work->shared_prompt_tokens`. In the current code, since that field stays zero for this manual workload, the paged backend will treat the shared prefix length as 0 (unless the compiler or some earlier version changed that). The comments indicate they intended to reflect shared prefix usage, but in this exact snapshot sharing may effectively be disabled by that mismatch.

In contrast, the `generate_workload` function, if used, does set `shared_prompt_tokens`. So there is a bit of divergence between the manual workload and the paged backend’s expectation.

### Running both backends

```c
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
```

- Creates monolithic backend and measures its stats.
- Then creates the paged backend and measures that.
- Prints both.

The README output:

```text
bytes_per_token = 8192
Monolithic:
  logical_bytes  = 621674496
  physical_bytes = 4294967296
  waste_bytes    = 3673292800 (85.53%)
Paged+Prefix:
  logical_bytes  = 621674496
  physical_bytes = 629800960
  waste_bytes    = 8126464 (1.29%)
```

Interpretation:

- `logical_bytes` are the same for both: same workload and model size.
- Monolithic `physical_bytes` is 4 GB, with about 85% waste.
- Paged+Prefix `physical_bytes` is only ~630 MB; extremely close to logical usage, ~1.3% overhead.

This demonstrates the key claim: **paged KV + prefix sharing** can give near-ideal memory efficiency, while monolithic KV can be extremely wasteful.

---

## Build system and layout

- Makefile compiles all `.c` files under src with `-O2 -Wall -std=c11 -pthread -Iinclude`.
- Target binary: `llm_sim`.
- `make clean` removes the binary and all `src/*.o`.

.gitignore ignores standard build artifacts and also the Codacy instructions.

---

## Summary of behavior

1. **Configuration**: `SimConfig` chooses model size, page size, arena size, and workload distribution.
2. **Workload**: `SequenceWork` describes each sequence’s prompt and generation lengths and sharing group.
3. **Simulation**:
   - `run_simulation` spins up `num_sequences` threads.
   - Each thread initializes a sequence via the backend and then appends `prompt_tokens + gen_tokens` tokens.
   - Sequences are kept alive until backend destroy, approximating peak memory.
4. **Monolithic backend**:
   - Each sequence gets a fixed-size buffer for 2048 tokens.
   - Physical memory = (#sequences) × (2048 tokens) × (bytes_per_token).
   - Very wasteful if average usage is far below 2048 tokens.
5. **Paged backend**:
   - Allocates large arena once and slices it into pages.
   - Shared prefixes allocated per group and shared among sequences by ref-counted pages.
   - Additional tokens allocate pages on demand.
   - Uses page counts and page size to compute physical bytes.
   - Yields much lower physical usage, especially when many sequences share large prefixes.

If you’d like, I can:

- Walk through an example numeric computation (e.g., from the README numbers).
- Or sketch how you could modify main.c to use `generate_workload` and wire `shared_prompt_tokens` correctly to see the effect of partial sharing + extra prompt tokens.