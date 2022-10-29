import struct

int_list = [0, 1, 258, 32768, 898123213]
fmt = "<%dI" % len(int_list)
data = struct.pack(fmt, *int_list)

fmt = "<I<3H<f"

