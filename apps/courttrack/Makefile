# Court tracker (C++ port of tools/world_tracker.py) and its replay harness.
CXX ?= g++
# -ffp-contract=off: no fused multiply-add (ARM), so results match the Python reference bit for bit
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -fPIC -ffp-contract=off

all: ct_replay libcourttrack.a

libcourttrack.a: courttrack.o online_teams.o
	ar rcs $@ $^

ct_replay: ct_replay.o libcourttrack.a
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp courttrack.h online_teams.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *.a ct_replay

.PHONY: all clean
