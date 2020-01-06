import numpy as np
class XXZmodel(object):
  #Defines the model - Hyperbolic reparametrized XXZ
  def __init__(self,levels_):
    self.levels=np.array(sorted(levels_))
    self.nlevels=len(self.levels)
  def Z(self,i,j):
    #Defines the Z-elements of the Gaudin algebra
    assert(j!=i)
    return (self.levels[i]+self.levels[j])/(self.levels[i]-self.levels[j])
  def Z_ia(self,i,x):
    return (self.levels[i]+x)/(self.levels[i]-x)
  def Z_ab(self,a,b):
    return (a+b)/(a-b)
  def X(self,i,j):
    #Defines the X-elements of the Gaudin algebra
    assert(j!=i)
    return 2.*np.sqrt(self.levels[i]*self.levels[j])/(self.levels[i]-self.levels[j])
  def get_c(self):
    #Also known as Gamma
    return -1.
  def get_nlevels(self):
    #Number of levels
    return self.nlevels
  def get_eigenvalues(self,var,g):
    #Returnes eigenvalues of the constants of motion
    return [0.5*(-1.-var[i]+0.5*g*np.sum([self.Z(i,k) for k in range(len(var)) if k!=i])) for i in range(len(var))]
  def invert(self,glam,g):
    #Quick method for obtaining the rapidities as the roots of a polynomial, fails at large number of excitations
    N=int(round(-np.sum(glam)))/2
    B=[0.5*(N-glam[i]/g)*self.levels[i]**(N-1) for i in range(N)]
    A=np.zeros([N,N])
    for i in range(N):
      for m in range(N):
        A[i][m]=(0.5*(glam[i]/g+N)-m)*self.levels[i]**(m-1)
     
    P=np.dot(np.linalg.inv(A),B)
    coef=np.zeros(N+1)
    coef[0]=1
    for i in range(1,N+1):
      coef[i]=P[N-i]
    return np.roots(coef)

  def evaluateRG(self,rap,g):
    #Evaluate the RG equations
    N=len(rap)
    return [1.+0.5*g*np.sum([(eps+rap[a])/(eps-rap[a]) for eps in self.levels])-g*np.sum([(rap[b]+rap[a])/(rap[b]-rap[a]) for b in range(N) if b!=a]) for a in range(N)]

  #Three functions necessary for obtaining the rapidities
  def F_lag(self,z,g,N):
    return 1./g-self.nlevels/2.-N+1.+np.sum([eps/(eps-z) for eps in self.levels])
  def G_lag(self,z,lam,N):
    return 0.5*np.sum([(lam[i]+N)/(self.levels[i]-z) for i in range(self.nlevels)])
  def H_lag(self,z):
    return z


class XXXmodel(object):
  #Defines the model - Rational XXX
  def __init__(self,levels_):
    self.levels=np.array(sorted(levels_))
    self.nlevels=len(self.levels)
  def Z(self,i,j):
    assert(j!=i)
    return 1./(self.levels[i]-self.levels[j])
  def X(self,i,j):
    assert(j!=i)
    return 1./(self.levels[i]-self.levels[j])
  def Z_ia(self,i,a):
    return 1./(self.levels[i]-a)
  def Z_ab(self,a,b):
    return 1./(a-b)
  def get_c(self):
    return 0.
  def get_nlevels(self):
    return self.nlevels
  def get_eigenvalues(self,var,g):
    return [0.5*(-1.-var[i]+0.5*g*np.sum([self.Z(i,k) for k in range(len(var)) if k!=i])) for i in range(len(var))]
  def invert(self,glam,g):
    N=int(round(-np.sum(glam)))/2
    B=[N*self.levels[i]**(N-1)-glam[i]/g*self.levels[i]**N for i in range(N)]
    A=np.zeros([N,N])
    for i in range(N):
      for m in range(N):
        A[i][m]=(glam[i]/g*self.levels[i]-m)*self.levels[i]**(m-1)
    P=np.dot(np.linalg.inv(A),B)
    coef=np.zeros(N+1)
    coef[0]=1
    for i in range(1,N+1):
      coef[i]=P[N-i]
    return np.roots(coef)
  def evaluateRG(self,rap,g):
    N=len(rap)
    return [1.+0.5*g*np.sum([1./(eps-rap[a]) for eps in self.levels])-g*np.sum([1./(rap[b]-rap[a]) for b in range(N) if b!=a]) for a in range(N)]
  def F_lag(self,z,g,N):
    return 2./g+np.sum([1/(eps-z) for eps in self.levels])
  def G_lag(self,z,lam,N):
    return np.sum([(lam[i])/(self.levels[i]-z) for i in range(self.nlevels)])
  def H_lag(self,z):
    return 1.



class XXZmodelHyp(object):
  #Defines the model - Hyperbolic XXZ
  def __init__(self,levels_):
    self.levels=np.array(sorted(levels_))
    self.nlevels=len(self.levels)
  def Z(self,i,j):
    assert(j!=i)
    return 1./np.tanh(self.levels[i]-self.levels[j])
  def Z_ia(self,i,x):
    return 1./np.tanh(self.levels[i]-x)
  def Z_ab(self,a,b):
    return 1./np.tanh(a-b+0.j)
  def X(self,i,j):
    assert(j!=i)
    return 1./np.sinh(self.levels[i]-self.levels[j])
  def get_c(self):
    return -1.
  def get_nlevels(self):
    return self.nlevels
  def get_eigenvalues(self,var,g):
    return [0.5*(-1.-var[i]+0.5*g*np.sum([self.Z(i,k) for k in range(len(var)) if k!=i])) for i in range(len(var))]
  def invert(self,glam,g):
    r=.5
    Z_ri=[-self.Z_ia(i,r) for i in range(self.nlevels)]
    N=int(round(-np.sum(glam)))/2
    B=[-N*self.get_c()*Z_ri[i]**(N-1)+glam[i]/g*Z_ri[i]**N for i in range(N)]
    A=np.zeros([N,N])
    for i in range(N):
      for m in range(N):
        A[i][m]=m*self.get_c()*Z_ri[i]**(m-1)-glam[i]/g*Z_ri[i]**m+(m-N)*Z_ri[i]**(m+1)
    P=np.dot(np.linalg.inv(A),B)
    coef=np.zeros(N+1)
    coef[0]=1
    for i in range(1,N+1):
      coef[i]=P[N-i]
    Z_ralpha=np.roots(coef)
    #print Z_ralpha
    return Z_ralpha
    return np.array([r-np.arctanh(1./zra+0.j) for zra in Z_ralpha])


class XXZmodelTrig(object):
  #Defines the model - Trigonometric XXZ
  def __init__(self,levels_):
    self.levels=np.array(sorted(levels_))
    self.nlevels=len(self.levels)
  def Z(self,i,j):
    assert(j!=i)
    return 1./np.tan(self.levels[i]-self.levels[j])
  def Z_ia(self,i,x):
    return 1./np.tan(self.levels[i]-x)
  def Z_ab(self,a,b):
    return 1./np.tan(a-b)
  def X(self,i,j):
    assert(j!=i)
    return 1./np.sin(self.levels[i]-self.levels[j])
  def get_c(self):
    return 1.
  def get_nlevels(self):
    return self.nlevels
  def get_eigenvalues(self,var,g):
    return [0.5*(-1.-var[i]+0.5*g*np.sum([self.Z(i,k) for k in range(len(var)) if k!=i])) for i in range(len(var))]
  def invert(self,glam,g):
    r=0.
    Z_ri=[-self.Z_ia(i,r) for i in range(self.nlevels)]
    N=int(round(-np.sum(glam)))/2
    B=[-N*self.get_c()*Z_ri[i]**(N-1)+glam[i]/g*Z_ri[i]**N for i in range(N)]
    A=np.zeros([N,N])
    for i in range(N):
      for m in range(N):
        A[i][m]=m*self.get_c()*Z_ri[i]**(m-1)-glam[i]/g*Z_ri[i]**m+(m-N)*Z_ri[i]**(m+1)
    P=np.dot(np.linalg.inv(A),B)
    coef=np.zeros(N+1)
    coef[0]=1
    for i in range(1,N+1):
      coef[i]=P[N-i]
    Z_ralpha=np.roots(coef)
    rap= np.array([r-np.arctan(1./zra+0.j) for zra in Z_ralpha])
    return rap




