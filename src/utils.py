import time
import sys
import numpy as np
from scipy.stats import norm

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
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def varsizes(allitems):
  for name, size in sorted(((name, sys.getsizeof(value)) for name, value in allitems),
                          key= lambda x: -x[1])[:10]:
      print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
