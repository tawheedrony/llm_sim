#ifndef PAGE_ALLOC_H
#define PAGE_ALLOC_H

#include <stddef.h>
#include "sim_config.h"

typedef struct Page Page;
typedef struct PageAllocator PageAllocator;

PageAllocator* page_allocator_create(const SimConfig* cfg);
void           page_allocator_destroy(PageAllocator* pa);

Page*  page_alloc(PageAllocator* pa);
void   page_inc_ref(PageAllocator* pa, Page* p);
void   page_dec_ref(PageAllocator* pa, Page* p);

size_t page_allocator_pages_in_use(PageAllocator* pa);
size_t page_allocator_page_bytes(PageAllocator* pa);

#endif
