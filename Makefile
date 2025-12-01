CFLAGS = -O2 -Wall -std=c11 -pthread -Iinclude
LDFLAGS = -pthread

OBJ = src/main.o src/sim.o src/mono_kv.o src/page_kv.o src/page_alloc.o src/workload.o

llm_sim: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

clean:
	rm -f llm_sim $(OBJ)