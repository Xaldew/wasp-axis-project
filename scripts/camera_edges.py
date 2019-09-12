#!/usr/bin/env python

import numpy as np

C = {
    0: [-30,  30, 10],
    1: [-60,  -5, 10],
    2: [-26, -18, 50],
    3: [+16,  50,  0],
}

cnt = 0
for i in range(4):
    for j in range(i + 1, 4):
        ec = np.array(C[j]) - np.array(C[i])
        dc = np.linalg.norm(ec)
        print("e{0} = c{1} - c{2} = {3}, |e{0}| = {4:.1f}".format(cnt, j, i, ec, dc))
        cnt += 1
