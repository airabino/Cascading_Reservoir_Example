import sys
import time
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import pyomo.environ as pyomo

from copy import copy,deepcopy
from itertools import chain
from scipy.stats import binom

from .utilities import IsIterable,FullFact

def BlendingProblemExample(volume,abv,data,solver_kwargs):

	#Pulling the keys from the data dict
	keys=data.keys()

	#Initializing the model as a concrete model (as in one that has fixed inputted values)
	model=pyomo.ConcreteModel()

	#Adding the variables for optimization
	model.x=pyomo.Var(keys,domain=pyomo.NonNegativeReals)

	#Adding the objective function
	model.cost=pyomo.Objective(expr=sum(model.x[c]*data[c]['cost'] for c in keys))

	#Adding constraints on volume and abv targets
	model.volume=pyomo.Constraint(expr=volume==sum(model.x[c] for c in keys))
	model.abv=pyomo.Constraint(expr=0==sum(model.x[c]*(data[c]['abv']-abv) for c in keys))

	#Initializing the solver
	solver=pyomo.SolverFactory(**solver_kwargs)

	#Solving the model
	solver.solve(model)

	return model

def CHRP_Objective(model,inputs):

	generation_cost=sum(
		inputs['c_g'][r,t]*model.u_g[r,t] for r in model.R for t in model.T)

	pumping_cost=sum(
		inputs['c_p'][r,t]*model.u_p[r,t] for r in model.R for t in model.T)

	transmission_cost=sum(
		inputs['c_t'][t]*model.u_t[t] for t in model.T)

	model.objective=pyomo.Objective(
		expr=generation_cost+pumping_cost+transmission_cost)

	return model

def CHRP_COE(model,inputs):

	generation_power=sum(
		model.u_g[r,t] for r in model.R for t in model.T)

	pumping_power=sum(
		model.u_g[r,t] for r in model.R for t in model.T)

	transmission_power=sum(
		model.u_t[t] for t in model.T)

	demand_power=sum(
		inputs['y_e'][t] for t in model.T)

	model.coe=pyomo.Constraint(
		expr=generation_power-pumping_power+transmission_power-demand_power==0)

	return model

def CHRP_Bounds(model,inputs):

	model.bounds=pyomo.ConstraintList()
	model.fs=pyomo.ConstraintList()

	for r in range(inputs['N']):
		level=inputs['is'][r]

		for t in range(inputs['M']):

			level+=(
				model.u_p[r,t]-
				model.u_g[r,t]+
				inputs['y_f'][r,t])

			model.bounds.add((inputs['lb'][r],level,inputs['ub'][r]))

		model.fs.add(expr=level==inputs['fs'][r])

	return model

def CHRP_Unit_Commitment(model,inputs):

	model.unit_commitment=pyomo.ConstraintList()

	for r in range(inputs['N']):
		for t in range(inputs['M']):

			model.unit_commitment.add(
				expr=inputs['f_g_min'][r]*model.u_gc[r,t]-model.u_g[r,t]<=0)
			
			model.unit_commitment.add(
				expr=inputs['f_g_max'][r]*model.u_gc[r,t]-model.u_g[r,t]>=0)

			model.unit_commitment.add(
				expr=inputs['f_p_min'][r]*model.u_pc[r,t]-model.u_p[r,t]<=0)
			
			model.unit_commitment.add(
				expr=inputs['f_p_max'][r]*model.u_pc[r,t]-model.u_p[r,t]>=0)

	return model

def CHRP_Variables(model,inputs):

	# generator_lb=np.tile(inputs['f_g_min'],(inputs['M'],1))
	# generator_ub=np.tile(inputs['f_g_max'],(inputs['M'],1))
	# pump_lb=np.tile(inputs['f_p_min'],(inputs['M'],1))
	# pump_ub=np.tile(inputs['f_p_max'],(inputs['M'],1))

	model.R=pyomo.Set(initialize=[idx for idx in range(inputs['N'])])
	model.T=pyomo.Set(initialize=[idx for idx in range(inputs['M'])])

	model.u_g=pyomo.Var(model.R,model.T,domain=pyomo.NonNegativeReals)
		# bounds=lambda m,r,t: (generator_lb[r,t],generator_ub[r,t]))

	model.u_p=pyomo.Var(model.R,model.T,domain=pyomo.NonNegativeReals)
	model.u_t=pyomo.Var(model.T,domain=pyomo.Reals,bounds=(0,inputs['t_max']))
	model.u_gc=pyomo.Var(model.R,model.T,domain=pyomo.Boolean)
	model.u_pc=pyomo.Var(model.R,model.T,domain=pyomo.Boolean)

	return model

def CHRP(inputs):

	#Pulling the keys from the inputs dict
	keys=inputs.keys()

	#Initializing the model as a concrete model (as in one that has fixed inputted values)
	model=pyomo.ConcreteModel()

	#Adding variables
	model=CHRP_Variables(model,inputs)

	#Adding the objective function
	model=CHRP_Objective(model,inputs)

	#Conservation of energy constraint
	model=CHRP_COE(model,inputs)

	#Bounds constraints
	model=CHRP_Bounds(model,inputs)

	#Unit commitment constraints
	model=CHRP_Unit_Commitment(model,inputs)

	return model

def unit_commitment():
    m = pyomo.ConcreteModel()

    m.N = pyomo.Set(initialize=N)
    m.T = pyomo.Set(initialize=T)

    m.x = pyomo.Var(m.N, m.T, bounds = (0, pmax))
    m.u = pyomo.Var(m.N, m.T, domain=pyo.Binary)
    
    # objective
    m.cost = pyomo.Objective(
    	expr = sum(m.x[n,t]*a[n] * m.u[n,t]*b[n] for t in m.T for n in m.N),
    	sense=pyo.minimize)
    
    # demand
    m.demand = pyomo.Constraint(m.T, rule=lambda m, t: sum(m.x[n,t] for n in N) == d[t])
    
    # semi-continuous
    m.lb = pyomo.Constraint(m.N, m.T, rule=lambda m, n, t: pmin*m.u[n,t] <= m.x[n,t])
    m.ub = pyomo.Constraint(m.N, m.T, rule=lambda m, n, t: pmax*m.u[n,t] >= m.x[n,t])
    return m