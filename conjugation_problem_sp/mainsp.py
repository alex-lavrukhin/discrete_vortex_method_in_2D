#!/usr/bin/python
import numpy as np
from sys import path
path.append("./../")
from smodule.sfunctp import *
Kx=1
Hx=1
Px=1
Ky=1
Hy=1
Py=1
Phi1=-0.25/sp.pi*sp.log((x0-y0)**2+(x1-y1)**2)-0.25/sp.pi*sp.log((x0-y0)**2+(x1+y1)**2)
Psi2=0.25/sp.pi*sp.log((x0-y0)**2+(x1-y1)**2)-0.25/sp.pi*sp.log((x0-y0)**2+(x1+y1)**2)

n=800
q=np.array([np.pi,np.pi])
z=np.array([[0.0,3.0],[3.0,2.0]],dtype='d')
lambdaK=0.8
minHDist=0.01
hv=0.2 # for velocity field grid
y_crcl=1.5
thet=np.linspace(2*np.pi,0,n+1,endpoint=True)
 
x=np.array([np.cos(thet),y_crcl+np.sin(thet)])
A,f=ieFredgolm2dk(x,x,lambdaK,0.0,q,z,varphiSources,Py,Phi1)
g=np.linalg.solve(A,f)
xs=np.mgrid[-4.:4.0001:hv,0.0001:4.0001:hv]
xs=xs[:,((xs[0]-z[0][0])**2+(xs[1]-z[1][0])**2)>0.4]
xs=xs[:,((xs[0]-z[0][1])**2+(xs[1]-z[1][1])**2)>0.4]
Ws=VSources(q,xs,z,Kx,Phi1)+VDL(xs,x,g,Hx,Psi2)
from pylab import quiver,show, gca,Circle,text, axis, grid
gca().add_patch(Circle((0,y_crcl),radius=1,alpha =.5, fc='y'))
quiver(xs[0],xs[1],Ws[0],Ws[1],color='r')
"""
# Testing on the exact solution
qOut=np.array([np.pi,lambdaK*np.pi,-lambdaK*np.pi])
zOut=np.array([[0.0,0.0,0.0],[2.0,0.5,0.0]],dtype='d')
xOut=xs[:,(xs[0]**2+xs[1]**2)>(1+minHDist)**2]
WOut=VSources(qOut,xOut,zOut)
WNOut=VSources(q,xOut,z)+VDL(xOut,x,g)
etaOut=(1.-np.sqrt(WNOut[0]**2+WNOut[1]**2)/np.sqrt(WOut[0]**2+WOut[1]**2))*100
etaOutMax=np.max(etaOut)
print etaOut
text(-2.5,3.1,ur"$\eta_{Out,max}=%6.4f$per." % etaOutMax)

qIn=np.array([(1-lambdaK)*np.pi])
xIn=xs[:,(xs[0]**2+xs[1]**2)<(1-minHDist)**2]
WIn=VSources(qIn,xIn,z)
WNIn=VSources(q,xIn,z)+VDL(xIn,x,g)
etaIn=(1-np.sqrt(WNIn[0]**2+WNIn[1]**2)/np.sqrt(WIn[0]**2+WIn[1]**2))*100
etaInMax=np.max(etaIn)
text(0,3.1,ur"$\eta_{In,max}=%6.4f$per." % etaInMax)
"""
#quiver(xOut[0],xOut[1],WOut[0],WOut[1],color='b',lw=1)
#quiver(xIn[0],xIn[1],WIn[0],WIn[1],color='g',lw=1)
#quiver(np.r_[xOut[0],xIn[0]],np.r_[xOut[1],xIn[1]],np.r_[WOut[0],WIn[0]],np.r_[WOut[1],WIn[1]])
#np.savetxt('W.dat',np.c_[xs[0].ravel(),xs[1].ravel(),Ws[0],Ws[1]],fmt='%6.4f',delimiter=' ')
grid(True)
axis('equal')
show()
