import numpy as np
import pylab as pl

from XXZmodels import *
from newSolver import *


if __name__ == '__main__':

  print("Small example for a ten-level picket fence model")

  levels=[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]    #Define energy levels epsilon_i
  XXZ=XXXmodel(levels)                       #Define the model, alternative XXZ=XXZmodel(levels) results in hyperbolic model leading to p+ip-pairing

  g=-1.                        #Define the coupling constant
  N_ex=5                       #Define the number of excitations
  eq=equations(XXZ,g,N_ex)     #Define equations object

  dist_init=[1,1,1,1,1,0,0,0,0,0]        #Distribution of the excitations over the levels at zero-coupling

  eq.solve(init=dist_init)            #Solve the substituted equations
  sol=eq.get_solution()               #Returns the set of g*Lambda_i
  error=eq.evaluate(sol)              #Evaluate the substituted equations  

  print "g*Lambda_i:\n%s"%sol)
  print("Error: %s"%np.linalg.norm(error))


  occ=eq.get_occupation()            #Get occupation numbers <S_i^0>
  print("Occupation numbers:\n%s"%occ)



  """
  Get the evolution of the variables for changing coupling constant
  """
  [g_evol,var_evol]=eq.solve(evol=True)        #evol=True returns the solutions at each value of the coupling constant 0..g

  #Plot evolution of the variables
  
  pl.figure()
  pl.ylabel(r'$g\Lambda_i$')
  pl.xlabel(r'$g$')
  for i in range(len(levels)):
    pl.plot([g_evol[j] for j in range(len(g_evol))],[var_evol[j][i] for j in range(len(g_evol))])
  #pl.savefig('evolutionLambda.pdf')
  pl.show()
  
  """
  Obtain the rapidities
  """
  eq.solve(evol=True,movingGrid=True)    #No init=... given results in ground state of BCS model, movingGrid=True allows for the calculation of the rapidities
  rap=eq.get_rapidities()        #Get rapidities
  print("Rapidities:\n%s"%rap)

  print("Larger example of a picket fence model")
  levels=[1.*i for i in range(1,41)]
  XXZ=XXZmodel(levels)
  g=-1.
  N_ex=20
  eq=equations(XXZ,g,N_ex)
  [g_evol,var_evol]=eq.solve(g_step=5e-3,evol=True)
  sol=eq.get_solution()
  print("Lambda_i:\n%s"%sol)
  error=eq.evaluate(sol)
  print("Error: %s"%np.linalg.norm(error))



  
