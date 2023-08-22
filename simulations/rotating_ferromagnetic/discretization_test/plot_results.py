import numpy as np
import matplotlib.pyplot as plt

curvatures = """
 2.160321484893399
 2.4760206442061548
 2.5591846725844665
 2.6002569967921287
 2.6196146034745937
 2.6304948510413615
 2.6373003382192333
 2.6419068184592898
 2.645581155339181
 2.649526290441222
"""
curvatures = curvatures.split()
curvatures = np.array([float(curvature) for curvature in curvatures])

ns = """
  10
  20
  30
  40
  50
  60
  70
  80
  90
 100
"""
ns = ns.split()
ns = np.array([int(n) for n in ns])

plt.plot(ns,curvatures)
plt.show()

bestcurv = curvatures[-1]

relerr = (bestcurv - curvatures)/bestcurv

plt.scatter(ns[:-1],relerr[:-1])
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("relative error")
plt.grid()
plt.show()

print("relerr at N=80", relerr[7])