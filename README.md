# LLM KV Cache Paging Simulator
LLM Inference Simulator that proves that paging and prefix-sharing reduces the memory waste for concurrent decode operations.

> make clean
rm -f llm_sim src/main.o src/sim.o src/mono_kv.o src/page_kv.o src/page_alloc.o src/workload.o
> make llm_sim
cc -O2 -Wall -std=c11 -pthread -Iinclude   -c -o src/main.o src/main.c
cc -O2 -Wall -std=c11 -pthread -Iinclude   -c -o src/sim.o src/sim.c
cc -O2 -Wall -std=c11 -pthread -Iinclude   -c -o src/mono_kv.o src/mono_kv.c
cc -O2 -Wall -std=c11 -pthread -Iinclude   -c -o src/page_kv.o src/page_kv.c
cc -O2 -Wall -std=c11 -pthread -Iinclude   -c -o src/page_alloc.o src/page_alloc.c
cc -O2 -Wall -std=c11 -pthread -Iinclude   -c -o src/workload.o src/workload.c
cc -O2 -Wall -std=c11 -pthread -Iinclude -o llm_sim src/main.o src/sim.o src/mono_kv.o src/page_kv.o src/page_alloc.o src/workload.o -pthread
> ./llm_sim
bytes_per_token = 8192
Monolithic:
  logical_bytes  = 621674496
  physical_bytes = 4294967296
  waste_bytes    = 3673292800 (85.53%)
Paged+Prefix:
  logical_bytes  = 621674496
  physical_bytes = 629800960
  waste_bytes    = 8126464 (1.29%)
