import time
import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

def readcfg(filename):
        dir = os.path.realpath(__file__) #_file = current script
        dir=os.path.dirname(dir) 
        dir=os.path.dirname(dir)        #second call to get src/..

        yamlfile=os.path.join(dir,filename)

        with open(yamlfile, "r") as f:
                return yaml.safe_load(f)

def readargs(config, parser):

    parser.add_argument("-i", "--infile", help="Input file (.GeoPIXE)", type=os.path.abspath)
    parser.add_argument("-o", "--outdir", help="Output path", type=os.path.abspath)
    parser.add_argument("-s", "--submap", action='store_true', help="Export submap (.GeoPIXE)")
    parser.add_argument("-p", "--parse", action='store_true', help="Only export submap")
    parser.add_argument("-f", "--force", action='store_true', help="Force recalculation of all pixels/classes")
    parser.add_argument('-c', "--coords", nargs='+', type=int, help="Coordinates for submap as: x1 y1 x2 y2")

    args = parser.parse_args()

    if args.infile is not None:
        config['infile'] = args.infile

    if args.outdir is not None:
        config['outdir'] = args.outdir

    if args.submap:
        config['WRITESUBMAP'] = True

    if args.parse:
        config['PARSEMAP'] = True
    else:
        print("EXPORTING SUBMAPS ONLY")

    if args.force:
        config['FORCEPARSE'] = True
        config['FORCERED'] = True
        config['FORCEKMEANS'] = True

    if args.coords is not None:
        config['submap_x1']=args.coords[0]
        config['submap_y1']=args.coords[1]
        config['submap_x2']=args.coords[2]
        config['submap_y2']=args.coords[3]

        if not config['WRITESUBMAP']:
            print("WARNING: submap coordinates set but submap flag False")

    return config, args


def initcfg(config, args):
        
    if not config['PARSEMAP']:
        config['DOCOLOURS']=False
        config['DOCLUST']=False
        config['DOBG']=False

    if config['WRITESUBMAP']:
        if config['submap_x2'] == 0:
            config['submap_x2']=int(99999)
        if config['submap_y2'] == 0:
            config['submap_y2']=int(99999)

        if (config['submap_x1'] >= config['submap_x2']):
            raise ValueError("FATAL: x2 nonzero but smaller than x1")
        if (config['submap_y1'] >= config['submap_y2']):
            raise ValueError("FATAL: y2 nonzero but smaller than y1")
            
    script = os.path.realpath(__file__) #_file = current script
    spath=os.path.dirname(script) 
    spath=os.path.dirname(spath)
    
    #check if paths are absolute or relative based on leading /
    if config['infile'].startswith('/'):
        fi=config['infile']
    else:
        fi = os.path.join(spath,config['infile'])

    if config['outdir'].startswith('/'):
        odir=config['outdir']
    else:
        odir=os.path.join(spath,config['outdir'])

    #extract name of input file
    fname = os.path.splitext(os.path.basename(fi))[0]
    print(f"input file: {fi}")

    print(
        "---------------------------\n"
        "PATHS\n"
        "---------------------------\n"
        f"local: {spath}\n"
        f"data: {fi}\n"
        f"output: {odir}"
    )

    #check filetype is recognised - currently only accepts .GeoPIXE
    if not config['FTYPE'] == ".GeoPIXE":
        raise ValueError(f"FATAL: filetype {config['FTYPE']} not recognised")

    if config['WRITESUBMAP']:
        subname=fname+config['convext']
        fsub = os.path.join(odir,subname+config['FTYPE'])

        if not subname == os.path.splitext(os.path.basename(fsub))[0]:
            raise ValueError(f"submap name not recognisable")

        print(f"submap: {fsub}")
    else:
        fsub = None

    print("---------------------------")
    print("---------------------------")

    return config, fi, fname, fsub, odir

def lookfor(x, val):
    difference_array = np.absolute(x-val)
    index = difference_array.argmin()
    return index

def normgauss(x, mu, sig1, amp):
    """
    creates a gaussian along x
    normalised so max = amp
    """
    g1=norm.pdf(x, mu, sig1)
    g1n=np.divide(g1,max(g1))
    return np.multiply(g1n, amp)

def timed(f):
    """

    measures time to run function f
    returns tuple of (output of function), time
        WARNING: not sure what happens when f() itself returns tuple

    call as: 
        out, runtime=timed(lambda: gapfill2(data))
    
    https://stackoverflow.com/questions/5478351/python-time-measure-function
    """
    start = time.time()
    ret = f()
    elapsed = time.time() - start
    return ret, elapsed

def gapfill(x, y, nchannels):
    """
    fills gaps in function using dict
    
    basically assign dict of i,y pairs
        use dict to return default value of (i,0) if i not in dict

        kludge here - we only want (i,0) but *d fails if not given a (0,0) tuple
            .: give (i,(0,0)) but slice out first 0 only
        sure there is a better way to do this
    
    original:
        d = {k: v for k, *v in data}
        return([(i, *d.get(i, (0, 0))) for i in range(nchannels)])

    https://stackoverflow.com/questions/54724987/python-filling-gaps-in-list
    """
    d={}
    j=0
    for k in x:
                d[k] = (y[j],0)
                j+=1
    xout=np.zeros(nchannels,dtype=int)
    yout=np.zeros(nchannels, dtype=int)

    for i in range(nchannels):
        xout[i]=i
        yout[i]=(d.get(i, (0, 0))[0])
    return xout, yout

def sizeof_fmt(num, suffix='B'):
        ''' by Fred Cirera,    https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
                if abs(num) < 1024.0:
                        return "%3.1f %s%s" % (num, unit, suffix)
                num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

def varsizes(allitems):
    print(
        "---------------------------\n"
        "Memory usage:\n"
        "---------------------------\n"
    )
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in allitems),
                                                    key= lambda x: -x[1])[:10]:
            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def pxinsubmap(config, xcoord, ycoord):
    if (xcoord >= config['submap_x1'] and xcoord < config['submap_x2'] and
            ycoord >= config['submap_y1'] and ycoord < config['submap_y2']
    ):
        return True
    else:
        return False