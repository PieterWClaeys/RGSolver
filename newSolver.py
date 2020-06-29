import numpy as np
from XXZmodels import *
from laguerre import *
import pylab as pl
#from permutations import *

class equations(object):
  def __init__(self,XXZ_,kop_,excitations_,sol_ = None):
    """
    Set of equations in the new rescaled Lambda_i variables for general RG models
    """
    self.XXZ=XXZ_            #Defines the model - XXXmodel,XXZmodel,XXZmodelTrig or XXZmodelHyp
    self.levels=self.XXZ.levels        #Energy levels
    self.g = kop_            #Coupling constant
    self.gamma = self.XXZ.get_c()    #Gamma associated with Gaudin algebra
    self.N = excitations_        #Number of excitations
    self.n=self.XXZ.get_nlevels()    #Number of single-particle levels
    self.rapidities=None;        #Rapidities (have to be calculated)
    if sol_ == None:
      self.solution = None        #Set of g*Lambda_i (have to be calculated)
    else: 
      self.solution = np.array(sol_)
      assert(len(self.solution) == len(self.levels))
    assert(self.N <= self.n)
    self.occupation=None        #Set of occupation numbers (follow from derivative of g*Lambda_i)
  

  def get_solution(self):    #Return the set of g*Lambda_i
    return self.solution

  def get_occupation(self):    #Return the set of occupation numbers
    return self.occupation

  def get_rapidities(self):    #Return the set of rapidities
    return self.rapidities;
    
  def evaluate(self,var,g=None):
    """
    Evaluate the substituted equations for a set of variables 'var'=g*Lambda at coupling constant g
    """
    if (g==None):g=self.g
    assert(len(var)==self.n)
    res=np.zeros(self.n+1)
    for i in range(self.n):
      res[i]=var[i]**2+2.*var[i]-self.N*(self.n-self.N)*g**2*self.gamma-g*np.sum([self.XXZ.Z(i,j)*(var[i]-var[j]) for j in range(self.n) if j!=i])
    res[self.n]=np.sum(var)+2.*self.N
    return res


  def jacobian(self,var,g=None):
    """
    Calculate the Jacobian for the set of non-linear equations for a set of variables 'var' at coupling constant g
    """
    if (g==None):g=self.g
    jac=np.zeros([self.n+1,self.n])
    for i in range(self.n):
      for j in range(self.n):
        if(i==j): jac[i][j]=2.*(var[i]+1.)-g*np.sum([self.XXZ.Z(i,k) for k in range(self.n) if k!=i])
        else: jac[i][j]=g*self.XXZ.Z(i,j)
    for i in range(self.n):
      jac[self.n][i]=1.
    return jac

  def get_derivative(self,var,g=None):
    """
    Solve a set of linear equations for obtaining the derivatives to g of Lambda_i at coupling constant g starting from the set of Lambda_i
    """
    if (g==None):g=self.g
    A=np.zeros([self.n+1,self.n])
    B=np.zeros([self.n+1])
    for i in range(self.n):
      B[i]=self.gamma*2.*g*self.N*(self.n-self.N)+np.sum([self.XXZ.Z(k,i)*(var[k]-var[i]) for k in range(self.n) if k!=i])
      A[self.n][i]=1
      for j in range(self.n):
        if(i==j): A[i][j]=2.*var[i]+2.+g*np.sum([self.XXZ.Z(k,i) for k in range(self.n) if k!=i])
        else: A[i][j]=-g*self.XXZ.Z(j,i)
    Ainv=np.linalg.pinv(A)
    der=np.dot(Ainv,B)
    return der
  
  def taylor_expansion(self,g_temp,g_step,var):
    """
    Taylor expansion of the variables at coupling constant g_temp to g_temp+g_step, solves a set of linear equations for the derivatives
    """
    A=np.zeros([self.n+1,self.n])
    for i in range(self.n):
      A[self.n][i]=1
      for j in range(self.n):
        if(i==j): A[i][j]=2.*var[i]+2.+g_temp*np.sum([self.XXZ.Z(k,i) for k in range(self.n) if k!=i])
        else: A[i][j]=-g_temp*self.XXZ.Z(j,i)
    #First derivative
    B1=np.zeros(self.n+1)
    for i in range(self.n): 
      B1[i]=self.gamma*2.*g_temp*self.N*(self.n-self.N)+np.sum([self.XXZ.Z(k,i)*(var[k]-var[i]) for k in range(self.n) if k!=i])
    Ainv=np.linalg.pinv(A)
    der1=np.dot(Ainv,B1)
    #Second derivative
    B2=np.zeros(self.n+1)
    for k in range(self.n):
      B2[k]=self.gamma*2.*self.N*(self.n-self.N) -2.*der1[k]**2+2.*np.sum([self.XXZ.Z(l,k)*(der1[l]-der1[k]) for l in range(self.n) if k!=l])
    der2=np.dot(Ainv,B2)
    #Third derivative
    B3=np.zeros(self.n+1)
    for k in range(self.n):
      B3[k]=-6*der1[k]*der2[k]+3.*np.sum([self.XXZ.Z(l,k)*(der2[l]-der2[k]) for l in range(self.n) if k!=l])
    der3=np.dot(Ainv,B3)
    #Fourth derivative
    B4=np.zeros(self.n+1)
    for k in range(self.n):
      B4[k]=-8.*der3[k]*der1[k]-6.*der2[k]*der2[k]+4.*np.sum([self.XXZ.Z(l,k)*(der3[l]-der3[k]) for l in range(self.n) if k!=l])
    der4=np.dot(Ainv,B4)
 
    return var+g_step*der1+g_step**2*der2/2.+g_step**3*der3/6.+g_step**4*der4/24.

  def newtonraphson(self,g_temp,var_init):
    """
    Newton-Raphson method for solving the equations iteratively at fixed coupling constant
    """
    n_step=0
    error=np.linalg.norm(self.evaluate(var_init,g_temp))

    while (error > 1e-12 and n_step < 50):
      #Improve solution while error is too large and the number of steps does not exceed a limit
      J_inv=np.linalg.pinv(self.jacobian(var_init,g_temp))
      var_new=var_init-np.dot(J_inv,self.evaluate(var_init,g_temp))
      error=np.linalg.norm(self.evaluate(var_new,g_temp))
      var_init=var_new
      n_step+=1

    return var_init

  def solve(self,init=None,g_init=1e-3,g_step=5e-3,g_fin=None,evol=False,movingGrid=False):
    """
    Solve the equations iteratively starting from the weak-coupling limit
    Weak-coupling limit approximation at g=g_init, increases in steps g_steps untill g_fin is reacher
    evol=True returns the solutions at each value of g
    movingGrid=True calculates the rapidities (can be slow)
    """
    if(g_fin==None): g_fin=self.g
    #Check if all signs are correct
    if(g_fin<0):
      if(g_step>0): g_step*=-1.
      if(g_init>0): g_init*=-1.
    else:
      if(g_step<0): g_step*=-1.
      if(g_init<0): g_step*=-1.

    #If no initial distribution is given, start from the BCS ground state
    if(init==None): init=[1 if i<self.N else 0 for i in range(self.n)]
    var_init=np.array([-2.*init[i]-g_init/(1-2.*init[i])*np.sum([self.XXZ.Z(j,i)*(init[j]-init[i]) for j in range(self.n) if j!=i]) for i in range(self.n)])
    n_step=int((g_fin-g_init)/g_step)
    g=g_init

    #Define necessary variables if evol or movingGrid=True
    if(evol or movingGrid):
      var_evol=np.zeros([n_step,self.n])
      g_evol=np.zeros(n_step)
    if(movingGrid):
      rap_evol = np.zeros([n_step,self.N],dtype=complex)
      rap_evol[0] = [self.levels[i] for i in range(self.n) if init[i]!=0 ]
      rap=np.array([self.levels[i]+0.5*np.abs(np.random.rand()) for i in range(self.n) if init[i]!=0])
      grid=np.zeros(self.N+1,dtype=complex)
      grid[0]=1e3
      for k in range(self.N): grid[k+1]=rap[k]
      n_grid=n_step/20    #Calculates rapidities at 20 intermediate steps

    #Gradually increase the coupling constant g and solve iteratively at each step starting from the Taylor approximation from the previous step
    for i in range(n_step):
      var_new=self.newtonraphson(g,var_init)
      der=self.get_derivative(var_new,g)
      #var_init=self.taylor_expansion(g,g_step,var_new)
      var_init = var_new+g_step*der
      g+=g_step
      #print g

      #Save variables at current step if evol =True
      if(evol or movingGrid):
        var_evol[i]=var_init
        g_evol[i]=g
      if(movingGrid and i%n_grid==0 and i!=0):
        #Method for obtaining the rapidities starting from the set of Lambda_i
        rf=RootFinder(self.XXZ,var_evol[i]/g_evol[i],g_evol[i],self.N)
        u=rf.solveForU(grid)
        lm=LaguerreMethod(grid,u)
        rap=lm.laguerre()
        rap_evol[i]=np.sort(lm.laguerre())
        for k in range(self.N): grid[k+1]=rap[k]
        grid[0]=10*max(rap)
      elif(movingGrid and i!=0):
        rf=RootFinder(self.XXZ,var_evol[i]/g_evol[i],g_evol[i],self.N)
        u=rf.solveForU(grid)
        lm=LaguerreMethod(grid,u)
        rap_evol[i]=np.sort(lm.laguerre())
      
        
    #One final iterative solution at g=g_fin
    self.solution=self.newtonraphson(g_fin,var_init)
    #Calculate the occupation numbers
    self.occupation=0.5*(-1.-self.solution+g_fin*self.get_derivative(self.solution,g_fin))

    #One final calculation of the rapidities
    if(movingGrid):
      rf=RootFinder(self.XXZ,self.solution/g_fin,g_fin,self.N)
      u=rf.solveForU(grid)
      lm=LaguerreMethod(grid,u)
      rap=lm.laguerre()
      self.rapidities=rap

    if movingGrid: return [g_evol,var_evol,rap_evol]
    if evol: return [g_evol,var_evol]
    return self.solution




if __name__ == '__main__':
  #Example calculation for N excitations in L equally-spaced levels
  L = 6
  levels=[1.*i for i in range(1,L+1)]        #Defines energy levels
  XXX=XXXmodel(levels)                #Define rational model
  g=-10.                        #Coupling constant
  N_ex= 2                #Number of excitations
  eq=equations(XXX,g,N_ex)            #Define object containing the equations

  state_seed = [1 if i<N_ex else 0 for i in range(len(levels)) ]    #Define initial state
  print("Initial state: %s:"%state_seed)
  list_states = list(perm_unique(state_seed))
  dim = len(list_states)
  print("Dim = %s"%len(list_states))

  for state_init in list_states:
    [g_evol,var_evol,_]=eq.solve(init=state_init,g_init=1e-3,g_step=1e-2,evol=True,movingGrid=True)        #Solve and give results at each step of the coupling constant
    sol=eq.get_solution()                #Get solution
    rap=eq.get_rapidities()            #Get rapidities
    print("State: ", state_init, "\nRapidities: ", rap)
  
  """
  print("Solutions: \n%s"%sol)
  print("Rapidities: \n%s"%rap)
  print("Error: %s"%np.linalg.norm(eq.evaluate(sol)))
  print("Sum*(-0.5) should equal number of excitations: %s"%(-0.5*np.sum(sol)))

  
  occ=eq.get_occupation()
  print("Occupation numbers : \n%s"%occ)
  print("Sum should equal zero at half-filling: %s"%np.sum(occ))
  """





