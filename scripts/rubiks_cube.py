#!/usr/bin/env python3

"""Generate a Rubiks' Cube like scene."""

import random


bx,by,bz = 0, 0, 0
sp = 0.1
# colors = ["[0.9 0.9 0.9]", "[0 1 0]", "[1 0.55 0]", "[0 0 1]", "[1 0 0]", "[1 1 0]"]
# all_colors = colors * 6
# random.shuffle(all_colors)

def rand_color():
    return "[{0} {1} {2}]".format(random.random(), random.random(), random.random())


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
            """.format(bx + x + sp * x,
                       by + y + sp * y,
                       bz + z + sp * z,
                       rand_color()))
            cnt += 1
