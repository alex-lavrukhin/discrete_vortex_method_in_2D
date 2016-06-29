#!/usr/bin/python
import numpy as np
import pylab
from sys import path
path.append("./../")
from smodule.sfunctp import *
#Kx=x0**(2)
#Hx=1
#Px=x0**(2)
#Ky=1
#Hy=1
#Py=1
Kx = x1**(-2)
Px = x1**(-2)
Hx = 1
Ky = y1**(-2)
Py = y1**(-2)
Hy = 1
dt=0.000001		
T_max=0.0003

Phi1=x1*y1*((((x0-y0)**2+x1**2+y1**2)/(2*x1*y1))*(sp.log((((x0-y0)**2+(x1+y1)**2)**(0.5))/(((x0-y0)**2+(x1-y1)**2)**(0.5))))-1)/2/np.pi
Psi2=(-1)*(sp.log((((x0-y0)**2+(x1+y1)**2)**(0.5))/(((x0-y0)**2+(x1-y1)**2)**(0.5))))/(2*np.pi*x1*y1)

n=800
q=np.array([np.pi/500,np.pi/500])
z=np.array([[8.0,3.0],[3.0,2.0]],dtype='d')
lambdaK=0.8
minHDist=0.01
hv=0.2 # for velocity field grid

border4=np.genfromtxt(open('../borders/step_11.txt'),dtype='d')
Lk=np.array([border4[:,0],border4[:,1]]) 
pylab.plot(Lk[0],Lk[1],lw=3,color='g')

border3=np.genfromtxt(open('../borders/step_12.txt'),dtype='d')
x=np.array([border3[:,0],border3[:,1]]) 
pylab.plot(x[0],x[1],lw=3,color='darkred')

thet=np.linspace(2*np.pi,0,n+1,endpoint=True)
temp1=np.array([8+0.15*np.cos(thet),3+0.15*np.sin(thet)])
pylab.plot(temp1[0],temp1[1],lw=5,color='blue')
temp2=np.array([3+0.15*np.cos(thet),2+0.15*np.sin(thet)])
pylab.plot(temp2[0],temp2[1],lw=5,color='blue')

A,f=ieFredgolm2dk(x,x,lambdaK,0.0,q,z,varphiSources,Py,Phi1)
g=np.linalg.solve(A,f)
xs=np.mgrid[0.0001:16.0001:hv,0.0001:10.0001:hv]
xs=xs[:,((xs[0]-z[0][0])**2+(xs[1]-z[1][0])**2)>0.4]
xs=xs[:,((xs[0]-z[0][1])**2+(xs[1]-z[1][1])**2)>0.4]
Ws=VSources(q,xs,z,Kx,Phi1)+VDL(xs,x,g,Hx,Psi2)
from pylab import quiver,show, gca,Circle,text, axis, grid
quiver(xs[0],xs[1],Ws[0],Ws[1],color='r')

  # Euler's method
isMove=True
T=0
while isMove:
    Lk+=VSources(q,Lk,z,Kx,Phi1)+VDL(Lk,x,g,Hx,Psi2)
    T+=dt
    if abs(T-0.00015)<dt: pylab.plot(Lk[0],Lk[1],lw=1,color='g')   
    if T>=T_max:
        isMove=False
        pylab.plot(Lk[0],Lk[1],lw=3,color='g')
        

grid(True)
axis('equal')
show()
