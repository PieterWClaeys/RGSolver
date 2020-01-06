import numpy as np
from newSolver import *
from XXZmodels import *

class RootFinder(object):
  def __init__(self,XXZ_,lam_,g_,N_):
    self.XXZ=XXZ_				#Richardson-Gaudin model
    self.levels=self.XXZ.levels			#Energy levels
    self.n=len(self.levels)			#Number of levels
    self.lam=lam_				#Set of Lambda_i (NOT g*Lambda_i this time)
    self.g=g_ 					#Coupling constant
    self.N=N_					#Number of excitations


  #Three functions following from the differential equations for each polynomial
  def F(self,z):
    return self.XXZ.F_lag(z,self.g,self.N)

  def G(self,z):
    return self.XXZ.G_lag(z,self.lam,self.N)

  def H(self,z):
    return self.XXZ.H_lag(z)

  def solveForU(self,z):
    #Obtain a Lagrange representation for the polynomial defining the rapidities by solving a linear set of equations
    #z is an (N+1)-array with the positions at which we do the Lagrange interpolation
    A=np.zeros([self.N+2,self.N+1],dtype=complex)
    B=np.zeros([self.N+2],dtype=complex)
    B[self.N+1]=1.
    for i in range(self.N+1):
      A[self.N+1][i]=1.
      for j in range(self.N+1):
        if(i==j):
          A[i][j]=self.F(z[i])*np.sum([1./(z[i]-z[k]) for k in range(self.N+1) if k!=i])
          A[i][j]+=self.H(z[i])*np.sum([np.sum([1./((z[i]-z[k])*(z[i]-z[l])) for l in range(self.N+1) if l!=i and l!=k]) for k in range(self.N+1) if k!=i])
          A[i][j]-=self.G(z[i])
        else:
          A[i][j]=self.F(z[i])/(z[i]-z[j])
          A[i][j]+=2.*self.H(z[i])*np.sum([1./((z[i]-z[j])*(z[i]-z[k])) for k in range(self.N+1) if (k!=i and k!=j)])
    Ainv=np.linalg.pinv(A)
    u=np.dot(Ainv,B)
    return u
  
class LaguerreMethod(object):
  """
  Containg an implementation of the Laguerre method for finding the roots of a polynomial starting from a Lagrange representation of that polynomial
  """
  def __init__(self,z_,u_):
    """
    Representation of polynomial P(z) 
    """
    assert(len(z_)==len(u_))
    self.N=len(z_)-1
    self.zgrid=z_
    self.ugrid=u_
    #print "Sum u_i (should be 1): %s" % (np.sum(self.ugrid))
  
  def evaluate(self,z,ugrid=None,zgrid=None):
    #Evaluate the polynomial at z
    if(ugrid is None):ugrid=self.ugrid
    if(zgrid is None):zgrid=self.zgrid
    return np.sum([ugrid[i]/(z-zgrid[i]) for i in range(len(ugrid))])

  def evaluateder(self,z,ugrid,zgrid):
    #Evaluate the first derivative of the polynomial at z
    res=0.
    for i in range(len(ugrid)):
      for j in range(len(ugrid)):
        if(i!=j): res+=ugrid[i]/((z-zgrid[i])*(z-zgrid[j]))
    return res

  def evaluatesecder(self,z,ugrid,zgrid):
    #Evaluate the second derivative of the polynomial at z
    res=0.
    for i in range(len(ugrid)):
      for j in range(len(ugrid)):
        for k in range(len(ugrid)):
          if(i!=j and j!=k and k!=i): res+=ugrid[i]/((z-zgrid[i])*(z-zgrid[j])*(z-zgrid[k]))
    return res

  def laguerreSingle(self,ugrid,zgrid):
    """
    Laguerre method for obtaining a single root
    """    
    treshold=1e-12
    x0=min(zgrid)-0.2*np.abs(min(zgrid))		#Initial guess
    error=np.abs(self.evaluate(x0,ugrid,zgrid))		#Initial error

    n=len(ugrid)
    teller=0
    while(error > treshold and teller < 40):
      #Search root iteratively while error is too large and the number of steps does not exceed the limit
      #Actual Laguerre method
      G=self.evaluateder(x0,ugrid,zgrid)/self.evaluate(x0,ugrid,zgrid)
      H=G**2-self.evaluatesecder(x0,ugrid,zgrid)/self.evaluate(x0,ugrid,zgrid)
      wrt=np.sqrt((n-1.)*(n*H-G**2)+0.j)
      if(np.abs(G+wrt)>np.abs(G-wrt)):a=n/(G+wrt)
      else: a=n/(G-wrt)
      x0=x0-a
      error=np.abs(self.evaluate(x0,ugrid,zgrid))
      #print "Laguerre error: %s"% error
      teller+=1
    #print "Laguerre: %s - %s"%(teller,error)
    return x0

  def laguerre(self,x0=None):
    """
    Laguerre method for obtaining all roots
    """
    if(x0==None): x0=(min(self.zgrid.real)+1.)
    roots=np.zeros(self.N,dtype=complex)
    u_new=np.array(self.ugrid)
    z_new=np.array(self.zgrid)
    res=self.laguerreSingle(u_new,z_new)
    teller=0
    roots[teller]=res
    while(teller<self.N-1):
      #Search roots untill all roots are found
      #Update polynomial representation once a root is found
      pos=(np.abs(z_new-res)).argmin()
      u_upd=[u_new[i]*(z_new[i]-z_new[pos])/(z_new[i]-res) for i in range(len(z_new)) if i!=pos]
      z_upd=[z_new[i] for i in range(len(z_new)) if i!=pos]
      u_new=u_upd
      z_new=z_upd
      res=self.laguerreSingle(u_new,z_new)
      teller+=1
      roots[teller]=res
    #for r in roots: print self.evaluate(r)
    return roots




