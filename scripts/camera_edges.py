#!/usr/bin/env python

import json
import sys
import argparse
import locale
import numpy as np


def main(args):
    """Generate the edges in a fully connected camera graph.

    .. Returns:
    :returns: 0 if the script ran successfully, otherwise a non-zero value.
    :rtype: An integer.

    """
    cameras = json.load(args.input)
    vertices = list()
    C = dict()
    for i, (k, v) in enumerate(sorted(cameras.items())):
        C[i] = v
        img = "cam-%03d.jpg" % i
        vertices.append({"index": i, "image": img, "scene": k, "location": v})

    cnt = 0
    edges = list()
    for i in range(len(C)):
        for j in range(len(C)):
            if i == j:
                continue
            ec = np.array(C[j]) - np.array(C[i])
            dc = np.linalg.norm(ec)
            # print("e{0} = c{1} - c{2} = {3}, |e{0}| = {4:.1f}".format(cnt, j, i, ec, dc))
            cnt += 1
            edges.append([i, j, dc])

    data = {"vertices": vertices, "edges": edges}
    json.dump(data, args.output, indent=4, sort_keys=True)
    return 0


def parse_arguments(argv):
    """Parse the given argument vector.

    .. Keyword Arguments:
    :param argv: The arguments to be parsed.

    .. Types:
    :type argv: A list of strings.

    .. Returns:
    :returns: The parsed arguments.
    :rtype: A argparse namespace object.

    """
    fmtr = argparse.RawDescriptionHelpFormatter
    kdesc = "Camera edge calculator"
    parser = argparse.ArgumentParser(description=kdesc, formatter_class=fmtr)
    parser.add_argument("input", metavar="JSON", type=argparse.FileType("r"),
                        help="JSON file with camera locations.")
    parser.add_argument("output", metavar="JSON", type=argparse.FileType("w"),
                        help="JSON file with edge distances.")
    return parser.parse_args(argv)


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, "")
    ARGS = parse_arguments(sys.argv[1:])
    sys.exit(main(ARGS))
