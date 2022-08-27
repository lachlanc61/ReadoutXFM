import time


def timed(f):
  """
  https://stackoverflow.com/questions/5478351/python-time-measure-function
  measures time to run function f
  returns tuple of (output of function), time
    WARNING: not sure what happens when f() itself returns tuple

  run as: 
    out, runtime=timed(lambda: gapfill2(data))
  """
  start = time.time()
  ret = f()
  elapsed = time.time() - start
  return ret, elapsed

def gapfill(x, y, nchannels):
  """
  https://stackoverflow.com/questions/54724987/python-filling-gaps-in-list
  fills gaps in function using dict
  basically assign dict of i,y pairs
    use dict to return default value of (i,0) if i not in dict

    cludge here - we only want (i,0) but *d fails if not given a (0,0) tuple
      .: give (i,(0,0)) but slice out first 0 only
    sure there is a better way to do this
  
  original:
    d = {k: v for k, *v in data}
    return([(i, *d.get(i, (0, 0))) for i in range(nchannels)])

  """
  d={}
  for k in x:
      for v in y:
          d[k] = [v]
          y.remove(v)
          break 
  return([(i, d.get(i, (0, 0))[0]) for i in range(nchannels)])


x=[0,1,2,4]
y=[1,2,4,2]
data=[(0,1), (1,2), (2,4), (4,2)]
print("DATA",data)

print(x)
print(y)

#out, runtime=timed(lambda: gapfill(data, 20))
out, runtime=timed(lambda: gapfill(x,y, 20))
print("out",out)
print("TIME",runtime)
