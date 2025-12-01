#ifndef PAGED_KV_H
#define PAGED_KV_H
#include "kv_backend.h"
#include "sim_config.h"
KVBackend* create_paged_backend(const SimConfig* cfg);
#endif
