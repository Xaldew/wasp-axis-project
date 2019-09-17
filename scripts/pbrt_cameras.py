#!/usr/bin/env python3

"""Find all camera locations from a collection of PBRT scene files."""

import re
import sys
import json
import locale
import argparse


def main(args):
    """Extract the camera centers from a collection of PBRT scene files.

    .. Returns:
    :returns: 0 if the script ran successfully, otherwise a non-zero value.
    :rtype: An integer.

    """
    centers = dict()
    for f in args.input:
        for line in f:
            line = line.strip()
            if "LookAt" in line:
                lookat = [s for s in line.split(" ") if s]
                cam = [float(v) for v in lookat[1:4]]
                centers[str(f.name)] = cam
    json.dump(centers, args.output, sort_keys=True, indent=4)
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
    kdesc = "PBRT Scene Orbit Generator"
    parser = argparse.ArgumentParser(description=kdesc, formatter_class=fmtr)
    parser.add_argument("input", metavar="JSON", nargs="+",
                        type=argparse.FileType("r"),
                        help="PBRT scene files.")
    parser.add_argument("output", metavar="JSON", type=argparse.FileType("w"),
                        help="JSON file with edge distances.")
    return parser.parse_args(argv)


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, "")
    ARGS = parse_arguments(sys.argv[1:])
    sys.exit(main(ARGS))
