import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", help="Input file (.GeoPIXE)", type=os.path.abspath)
parser.add_argument("-o", "--output", help="Output path", type=os.path.abspath)
parser.add_argument("-s", "--submap", action='store_true', help="Export submap (.GeoPIXE)")
parser.add_argument("-ss", "--onlysub", action='store_true', help="Only export submap")
parser.add_argument("-f", "--force", action='store_true', help="Force recalculation of all pixels/classes")
parser.add_argument('-c', "--coords", nargs='+', type=int, help="Coordinates for submap as: x1 y1 x2 y2")

args = parser.parse_args()

print(args)

print(args.input)

if args.input is None:

    print("input not given")

#bool flags
#https://docs.python.org/dev/library/argparse.html#action
#https://stackoverflow.com/questions/8259001/python-argparse-command-line-flags-without-arguments
"""
arguments saved as object as args.argument
if long name given, uses that
if not then just uses flag

for an on/off flag, use "action='store_true' " to store a True bool value if present else False

"""






