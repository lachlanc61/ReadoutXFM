import struct 
import numpy as np

#-------------------------------------
#FUNCTIONS
#-----------------------------------

def binunpack(map, sformat):
    """
    parse binary data via struct.unpack
    takes:
        stream of bytes
        byte index
        format flag for unpack (currently accepts: <H <f <I )
    returns:
        value in desired format (eg. int, float)
        next byte index
    """

    if sformat == "<H":
        nbytes=2
    elif sformat == "<f":
        nbytes=4
    elif sformat == "<I":
        nbytes=4
    else:
        raise ValueError(f"ERROR: {sformat} not recognised by local function binunpack")
        exit(0)

    #if perfect end
    #   unpack then flag next chunk
    if map.idx == (len(map.stream)-nbytes):
        retval = struct.unpack(sformat, map.stream[map.idx:map.idx+nbytes])[0]
        map.idx=map.idx+nbytes
        map.nextchunk()
    #if end mid unpack
    #   unpack partial, get next chunk, unpack next partial and concat
    elif map.idx > (len(map.stream)-nbytes):
        remaining=(len(map.stream)-map.idx)
        partial1 = map.stream[map.idx:map.idx+remaining]
        map.idx=map.idx+remaining
        map.nextchunk()
        partial2 = map.stream[map.idx:map.idx+(nbytes-remaining)]
        map.idx=map.idx+(nbytes-remaining)
        #concat partials
        partial=partial1+partial2
        #WARNING: DOES NOT WORK YET
        # aim is to concat bytes, apparently don't behave like strings
        # need to convert to bytearray or similar
        # https://stackoverflow.com/questions/28130722/python-bytes-concatenation
        retval=struct.unpack(sformat, partial)[0]

    #if not at end
    #   unpack and increment index
    else:
        #struct unpack outputs tuple
        #want int so take first value
        retval = struct.unpack(sformat, map.stream[map.idx:map.idx+nbytes])[0]
        map.idx=map.idx+nbytes

    return(retval)