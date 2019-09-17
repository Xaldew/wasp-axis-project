#!/usr/bin/env bash

root=$(git rev-parse --show-toplevel)
scene=${1:?"No scene given."}
raytracer=${2:?"Path to RayTracer repo not given."}
fmt=${3:-"${root}/orbit/cam-%03d"}
steps=${4:-5}

dir=$(dirname ${fmt})
mkdir -p ${dir}

scene=$(realpath ${scene})
fmt=$(realpath ${fmt})

# { cd ${raytracer} && bash ./scripts/pbrt_scene_orbit.sh ${scene} ${fmt} ${steps}; }
python3 ${root}/scripts/pbrt_cameras.py ${dir}/*.pbrt ${dir}/camloc.json
python3 ${root}/scripts/camera_edges.py ${dir}/camloc.json ${dir}/camgraph.json