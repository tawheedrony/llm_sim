#ifndef MONO_KV_H
#define MONO_KV_H
#include "kv_backend.h"
#include "sim_config.h"
KVBackend* create_monolithic_backend(const SimConfig* cfg);
#endif
