#ifndef SIM_H
#define SIM_H
#include "kv_backend.h"
#include "sim_config.h"
#include "workload.h"

KVStats run_simulation(KVBackend* backend,
                       const SimConfig* cfg,
                       const SequenceWork* work);

#endif
