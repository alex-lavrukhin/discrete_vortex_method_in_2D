#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pylab
import sympy as sp
from numpy import genfromtxt
from sys import path

myeps=0.0000001
n=500
x0,x1,y0,y1=sp.symbols('x0 x1 y0 y1',real=True)
sx=[x0,x1]
sy=[y0,y1]
#Kx=1
#Py=1
Kx=x1**2
Py=x1**2
#Phi1NL=-0.25/sp.pi*sp.log((x0-y0)**2+(x1-y1)**2)
Phi1NL = sp.log(((x0-y0)**2+(x1+y1)**2)/((x0-y0)**2+(x1-y1)**2)) / (sp.pi*2*x1*y1)
omgNL=[ Py*sp.diff(Phi1NL,sy[j]) for j in range(2) ]
#Phi2NL=-0.5/sp.pi*sp.atan((x1-y1)/(x0-y0))
Phi2NL = sp.log(((x0-y0)**2+(x1+y1)**2)/((x0-y0)**2+(x1-y1)**2)) * (x0-y0)/(2*sp.pi*x1)
+ (sp.atan((x1-y1)/(x0-y0)) - sp.atan((x1+y1)/(x0-y0))) / (2*sp.pi)

V2NL=[sp.ratsimp(Kx*sp.diff(Phi2NL,sx[j])) for j in range(2)]
VSNL=[sp.ratsimp(Kx*sp.diff(Phi1NL,sx[j])) for j in range(2)]

def Normal(x):
    n=x[0].shape[0]-1
    return np.array([-x[1,1:]+x[1,:n],x[0,1:]-x[0,:n]])

def normal(x):
    nr=Normal(x)
    return nr/np.sqrt(nr[0]**2+nr[1]**2)

def withoutZero(sFunct):
    def wrapped(sExpress,M,minDist=myeps):
        gd=np.sum([(M[i,1]-M[i,0])**2 for i in range(2)],axis=0)>minDist**2
        AA=M[0,0].copy()
        AA[np.logical_not(gd)]+=1
        result=np.zeros_like(M[0,0])
        result[gd]=sFunct(sExpress,np.array([[AA,M[0,1]],[M[1,0],M[1,1]]]),minDist)[gd]
        return result
    wrapped.orig=sFunct
    return wrapped

@withoutZero
def sF(sExpress,M,minDist=myeps):
    return sp.lambdify((y0,x0,y1,x1),sExpress,modules='numpy')(M[0,0],M[0,1],M[1,0],M[1,1])

def MPDL(x,y,Ny,sOmega=omgNL,minDist=myeps):
    M=np.array([np.meshgrid(y[j],x[j]) for j in range(2)])
    return np.sum([sF(sOmega[j],M,minDist)*Ny[j] for j in range(2)],axis=0)

def PDL(x,y,g,Ny,sOmega=omgNL,minDist=myeps):
    return np.sum(MPDL(x,y,Ny,sOmega,minDist)*g,axis=1)

def Theta(M,rEps=myeps):
    R2=np.sum([(M[i,1]-M[i,0])**2 for i in range(2)],axis=0)
    THETA=np.ones_like(R2)
    rEps2=rEps**2
    R2=R2/rEps2; R=R2**.5; R3=R2*R; R5=R3*R2; R7=R5*R2; R9=R7*R2
    THETA[R2<rEps]=.125*(63.*R5-90.*R7+35.*R9)[R2<rEps]
    return THETA

def V2(x,y,sV2=V2NL,rEps=myeps):
    M=np.array([np.meshgrid(y[j],x[j]) for j in range(2)])
    return Theta(M,rEps)*np.array([sF(sV2[j],M) for j in range(2)])

def MVDL(x,b,sV2=V2NL,rEps=myeps):
    return -V2(x,b[:,1:],sV2,rEps)+V2(x,b[:,:b.shape[1]-1],sV2,rEps)

def VDL(x,b,g,sV2=V2NL,rEps=myeps):
    return np.sum(MVDL(x,b,sV2,rEps)*g,axis=2)

def varphiSources(q,x,z,sPhi1=Phi1NL):
    M=np.array([np.meshgrid(z[j],x[j]) for j in range(2)])
    return -np.sum(q*sF.orig(sPhi1,M),axis=1)

def VSources(q,x,z,sVS=VSNL):
    M=np.array([np.meshgrid(z[j],x[j]) for j in range(2)])
    return -np.array([np.sum(q*sF.orig(sVS[j],M),axis=1) for j in range(2)])

q=np.array([np.pi])
z=np.array([[3.0],[4.0]],dtype='d')
hv=0.1
minHDist=0.01
dt=0.03
R=1
radius_hole=0.08

#thet=np.linspace(2*np.pi,0,n+1,endpoint=True)
#x_i1=np.array([2+R*np.cos(thet),2+R*np.sin(thet)])

border1=genfromtxt(open('krivaya0_ot.txt'),dtype='d') 
x_i1=np.array([border1[:,0],border1[:,1]])
pylab.plot(x_i1[0],x_i1[1],lw=5)

xIc=(x_i1[:,:n]+x_i1[:,1:])/2.
Nxc=Normal(x_i1)
M=MVDL(xIc,x_i1,V2NL)
A=M[0]*Nxc[0]+M[1]*Nxc[1]
WS=VSources(q,xIc,z,VSNL)
f=-WS[0]*Nxc[0]-WS[1]*Nxc[1]

A=np.r_[np.c_[np.ones((n,1)),A],np.c_[0,np.ones((1,n))]]

f=np.r_[f,0]
g=np.linalg.solve(A,f)
gI=g[1:]

x_t=np.ones((n+1,2),dtype='d')
thet=np.linspace(2*np.pi,0,n+1,endpoint=True)
x_t=np.array([z[0]+R*np.cos(thet),z[1]+np.sin(thet)])
pylab.plot(x_t[0],x_t[1])

isMove=True
T=0
while isMove:
    x_t+=VSources(q,x_t,z,VSNL)*dt+VDL(x_t,x_i1,gI,V2NL)*dt
    T+=dt
    if abs(T-0.03)<dt: pylab.plot(x_t[0],x_t[1])
    if abs(T-0.06)<dt: pylab.plot(x_t[0],x_t[1]) 
    #if abs(T-2.5)<dt: pylab.plot(x_t[0],x_t[1]) 
    #if abs(T-0.56)<dt: pylab.plot(x_t[0],x_t[1]) 
    if T>0.09: isMove=False
    if np.logical_or.reduce( ( x_t[0]**2 + (x_t[1]-y1)**2  ) <= radius_hole**2 ):
        isMove=False
        print "T=%6.4f" % T
        
xs=np.mgrid[0.0:6.0:hv,0.0:6.0:hv]
xs=xs[:,((xs[0]-z[0])**2+(xs[1]-z[1])**2)>0.2**2]
xs=xs[:,np.logical_or((xs[0]**2+xs[1]**2)>(1.+minHDist)**2,(xs[0]**2+xs[1]**2)<(1.-minHDist)**2)]
Ws=VSources(q,xs,z,VSNL)+VDL(xs,x_i1,gI,V2NL)

# Testing on the exact solution
#qOut=np.array([np.pi,np.pi,-np.pi])
#zOut=np.array([[0.0,0.0,0.0],[2.0,0.5,0.0]],dtype='d')
#xOut=xs[:,(xs[0]**2+xs[1]**2)>(1+minHDist)**2]
#WOut=VSources(qOut,xOut,zOut,VSNL)
#WNOut=VSources(q,xOut,z,VSNL)+VDL(xOut,LI,g[1:],V2NL)
#etaOut=(1.-np.sqrt(WNOut[0]**2+WNOut[1]**2)/np.sqrt(WOut[0]**2+WOut[1]**2))*100
#etaOutMax=np.max(etaOut)

pylab.quiver(xs[0],xs[1],Ws[0],Ws[1],color='r')
#pylab.quiver(xOut[0],xOut[1],WOut[0],WOut[1],color='g')
pylab.grid(True)

pylab.show()
