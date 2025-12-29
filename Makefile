CFLAGS = -O2 -Wall -std=c11 -pthread -Iinclude
LDFLAGS = -pthread

SRC = src/main.c src/sim.c src/mono_kv.c src/page_kv.c src/page_alloc.c src/workload.c

llm_sim: $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f llm_sim