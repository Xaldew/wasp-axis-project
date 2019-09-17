.PHONY: all
all: debug release

.PHONY: debug
debug:
	mkdir --parents build/debug
	cd build/debug && cmake -DCMAKE_BUILD_TYPE=Debug ../..
	cd build/debug && $(MAKE)

.PHONY: release
debug:
	mkdir --parents build/release
	cd build/release && cmake -DCMAKE_BUILD_TYPE=Release ../..
	cd build/release && $(MAKE)
