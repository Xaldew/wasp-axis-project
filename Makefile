INKSCAPE ?= inkscape
IMGMAGICK ?= convert-im6

.PHONY: all
all: debug release images

.PHONY: debug
debug:
	mkdir --parents build/debug
	cd build/debug && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Debug ../..
	cd build/debug && $(MAKE)
	ln -f -s -T build/debug/compile_commands.json compile_commands.json

.PHONY: release
release:
	mkdir --parents build/release
	cd build/release && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release ../..
	cd build/release && $(MAKE)

.PHONY: images
images: $(wildcard report/*.svg)

.PHONY: clean
clean:
	$(RM) -r build
	$(RM) $(wildcard report/*.html report/*.pdf report/*.tex)


# Pattern rules
%.png: INK_FLAGS :=
%.png: %.svg
	$(INKSCAPE) "$<" --export-png="$@" $(INK_FLAGS)
%.pdf: %.svg
	$(INKSCAPE) "$<" --export-pdf="$@" $(INK_FLAGS)
%.ico: %.png
	$(IMGMAGICK) "$<" "$@"
