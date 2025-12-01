#define _GNU_SOURCE 1
#include <sys/mman.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include "sim_config.h"
#include <unistd.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

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

PageAllocator* page_allocator_create(const SimConfig* cfg) {
    PageAllocator* pa = (PageAllocator*) calloc(1, sizeof(PageAllocator));
    if (!pa) abort();

    pa->page_bytes = cfg->tokens_per_page * bytes_per_token(cfg);
    pa->num_pages  = cfg->arena_bytes / pa->page_bytes;

    size_t arena_size = pa->num_pages * pa->page_bytes;
    pa->arena = (unsigned char*) mmap(NULL, arena_size,
                                      PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pa->arena == MAP_FAILED) {
        free(pa);
        abort();
    }

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

void page_allocator_destroy(PageAllocator* pa) {
    size_t arena_size = pa->num_pages * pa->page_bytes;
    munmap(pa->arena, arena_size);
    free(pa->pages);
    free(pa->free_list);
    pthread_mutex_destroy(&pa->mutex);
    free(pa);
}

Page* page_alloc(PageAllocator* pa) {
    pthread_mutex_lock(&pa->mutex);
    if (pa->free_count == 0) {
        pthread_mutex_unlock(&pa->mutex);
        abort(); // out of pages in this simulation
    }
    Page* p = pa->free_list[--pa->free_count];
    p->ref = 1;
    pthread_mutex_unlock(&pa->mutex);
    return p;
}

void page_inc_ref(PageAllocator* pa, Page* p) {
    (void) pa;
    // for a simulator, we can just increment without atomic
    p->ref++;
}

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
