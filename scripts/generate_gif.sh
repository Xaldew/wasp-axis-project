#!/usr/bin/env sh

fmt=${1:?"No image format pattern given. (e.g., cam-%03d.jpg)."}
runtime=${2:-5}
dir=$(dirname ${fmt})

frames=0
while true; do
    f=$(printf ${fmt} ${frames})
    if [ ! -f ${f} ]; then
        break
    else
        frames=$((${frames} + 1))
    fi
done

# Generate GIF image.
framerate=$(( ${frames} / ${runtime}))
palette=$(mktemp XXXX_XXXX.png)
ffmpeg \
    -y \
    -loglevel error \
    -i ${fmt} \
    -filter:v palettegen \
    $palette \
    </dev/null
ffmpeg \
    -y \
    -loglevel error \
    -i ${palette} \
    -framerate ${framerate} \
    -f image2 \
    -i ${fmt} \
    -filter_complex "[1:v][0:v] paletteuse" \
    ${dir}/anim.gif \
    </dev/null
rm ${palette}
