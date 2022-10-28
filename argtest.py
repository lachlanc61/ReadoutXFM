import struct

int_list = [0, 1, 258, 32768]
fmt = "<%dI" % len(int_list)
data = struct.pack(fmt, *int_list)