#!/usr/bin/env python3

"""Generate a Rubiks' Cube like scene."""

import random


bx,by,bz = 0, 0, 0
colors = ["[0.9 0.9 0.9]", "[0 1 0]", "[1 0.55 0]", "[0 0 1]", "[1 0 0]", "[1 1 0]"]
all_colors = colors * 6
random.shuffle(all_colors)
cnt = 0
for x in range(0, 3):
    for y in range(0, 3):
        for z in range(0, 3):
            if x == y and y == z and z == 1: continue
            print("""
AttributeBegin
    Material "matte" "color Kd" {3}
    Translate {0} {1} {2}
    Shape "box" "float x" 1 "float y" 1 "float z" 1
AttributeEnd
            """.format(bx + x, by + y, bz + z, all_colors[cnt]))
            cnt += 1
