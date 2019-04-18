#!/usr/bin/env python3
import argparse
import jic_py



parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs='+')
args = parser.parse_args()


for fname in args.filenames:
	jic_py.get_run_dimensions(fname, echo=True)
