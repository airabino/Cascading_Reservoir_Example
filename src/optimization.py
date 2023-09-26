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

class CHRP():

	def __init__(self,inputs={}):

		self.inputs=inputs

		if self.inputs:

			self.Build()

	def Solve(self,solver_kwargs={}):

		solver=pyomo.SolverFactory(**solver_kwargs)
		solver.solve(self.model)

		self.solution=self.Solution()

	def Build(self):

		#Pulling the keys from the inputs dict
		keys=self.inputs.keys()

		#Initializing the model as a concrete model (as in one that has fixed inputted values)
		self.model=pyomo.ConcreteModel()

		#Adding variables
		self.Variables()

		#Adding the objective function
		self.Objective()

		#Conservation of energy constraint
		self.Conservation_of_Energy()

		#Bounds constraints
		self.Bounds()

		#Unit commitment constraints
		self.Unit_Commitment()

	def Variables(self):

		self.model.T=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_t'])])
		self.model.R=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_r'])])

		self.model.u_g=pyomo.Var(self.model.T,self.model.R,domain=pyomo.NonNegativeReals)

		self.model.u_p=pyomo.Var(self.model.T,self.model.R,domain=pyomo.NonNegativeReals)

		self.model.u_tp=pyomo.Var(self.model.T,domain=pyomo.Reals,
			bounds=(0,self.inputs['tp_max']))

		self.model.u_ts=pyomo.Var(self.model.T,domain=pyomo.Reals,
			bounds=(0,self.inputs['ts_max']))

		self.model.u_gc=pyomo.Var(self.model.T,self.model.R,domain=pyomo.Boolean)

		self.model.u_pc=pyomo.Var(self.model.T,self.model.R,domain=pyomo.Boolean)

	def Objective(self):

		transmission_purchase_cost=sum(
			self.inputs['c_tp'][t]*self.model.u_tp[t] for t in self.model.T)

		transmission_sell_cost=sum(
			self.inputs['c_ts'][t]*self.model.u_ts[t] for t in self.model.T)

		self.model.objective=pyomo.Objective(
			expr=transmission_purchase_cost+transmission_sell_cost)

	def Conservation_of_Energy(self):

		self.model.coe=pyomo.ConstraintList()

		for t in self.model.T:

			generation_power=sum(
				self.model.u_g[t,r] for r in self.model.R)

			pumping_power=sum(
				self.model.u_p[t,r] for r in self.model.R)

			transmission_purchase_power=self.model.u_tp[t]

			transmission_sell_power=self.model.u_ts[t]

			demand_power=self.inputs['y_e'][t]

			self.model.coe.add(
				expr=(generation_power-pumping_power+
					transmission_purchase_power-
					transmission_sell_power-
					demand_power==0))

	def Bounds(self):

		self.model.bounds=pyomo.ConstraintList()
		self.model.fs=pyomo.ConstraintList()

		for r in range(self.inputs['n_r']):
			level=self.inputs['is'][r]

			for t in range(self.inputs['n_t']):

				level+=(
					self.model.u_p[t,r]-
					self.model.u_g[t,r]+
					self.inputs['y_f'][t,r])

				self.model.bounds.add((self.inputs['lb'][r],level,self.inputs['ub'][r]))

			self.model.fs.add(expr=level==self.inputs['fs'][r])

	def Unit_Commitment(self):

		self.model.unit_commitment=pyomo.ConstraintList()

		for r in range(self.inputs['n_r']):
			for t in range(self.inputs['n_t']):

				self.model.unit_commitment.add(
					expr=(self.inputs['f_g_min'][r]*self.model.u_gc[t,r]-
						self.model.u_g[t,r]<=0))
				
				self.model.unit_commitment.add(
					expr=(self.inputs['f_g_max'][r]*self.model.u_gc[t,r]-
						self.model.u_g[t,r]>=0))

				self.model.unit_commitment.add(
					expr=(self.inputs['f_p_min'][r]*self.model.u_pc[t,r]-
						self.model.u_p[t,r]<=0))
				
				self.model.unit_commitment.add(
					expr=(self.inputs['f_p_max'][r]*self.model.u_pc[t,r]-
						self.model.u_p[t,r]>=0))

	def Solution(self):
		'''
		From StackOverflow
		https://stackoverflow.com/questions/67491499/
		how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
		'''
		model_vars=self.model.component_map(ctype=pyomo.Var)

		serieses=[]   # collection to hold the converted "serieses"
		for k in model_vars.keys():   # this is a map of {name:pyo.Var}
			v=model_vars[k]

			# make a pd.Series from each    
			s=pd.Series(v.extract_values(),index=v.extract_values().keys())

			# if the series is multi-indexed we need to unstack it...
			if type(s.index[0])==tuple:# it is multi-indexed
				s=s.unstack(level=1)
			else:
				s=pd.DataFrame(s) # force transition from Series -> df

			# multi-index the columns
			s.columns=pd.MultiIndex.from_tuples([(k,t) for t in s.columns])

			serieses.append(s)

		self.solution=pd.concat(serieses,axis=1)

class SCHRP():

	def __init__(self,inputs={}):

		self.inputs=inputs

		if self.inputs:

			self.Build()

	def Solve(self,solver_kwargs={}):

		solver=pyomo.SolverFactory(**solver_kwargs)
		solver.solve(self.model)

		self.solution=self.Solution()

	def Build(self):

		#Pulling the keys from the inputs dict
		keys=self.inputs.keys()

		#Initializing the model as a concrete model (as in one that has fixed inputted values)
		self.model=pyomo.ConcreteModel()

		#Adding variables
		self.Variables()

		#Adding the objective function
		self.Objective()

		#Conservation of energy constraint
		self.Conservation_of_Energy()

		#Bounds constraints
		self.Bounds()

		#Unit commitment constraints
		self.Unit_Commitment()

	def Variables(self):

		#Index sets
		self.model.T=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_t'])])
		self.model.R=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_r'])])
		self.model.S=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_s'])])

		#Specific variables
		self.model.u_g=pyomo.Var(self.model.T,self.model.R,self.model.S,
			domain=pyomo.NonNegativeReals)

		self.model.u_p=pyomo.Var(self.model.T,self.model.R,self.model.S,
			domain=pyomo.NonNegativeReals)

		self.model.u_gc=pyomo.Var(self.model.T,self.model.R,self.model.S,
			domain=pyomo.Boolean)

		self.model.u_pc=pyomo.Var(self.model.T,self.model.R,self.model.S,
			domain=pyomo.Boolean)

		self.model.u_tpd=pyomo.Var(self.model.T,self.model.S,
			domain=pyomo.Reals,
			bounds=(0,self.inputs['tpd_max']))

		self.model.u_tsd=pyomo.Var(self.model.T,self.model.S,
			domain=pyomo.Reals,
			bounds=(0,self.inputs['tsd_max']))

		#General variables
		self.model.u_tp=pyomo.Var(self.model.T,
			domain=pyomo.Reals,
			bounds=(0,self.inputs['tp_max']))

		self.model.u_ts=pyomo.Var(self.model.T,
			domain=pyomo.Reals,
			bounds=(0,self.inputs['ts_max']))

	def Objective(self):

		transmission_purchase_cost=0
		transmission_sell_cost=0
		transmission_purchase_deviation_cost=0
		transmission_sell_deviation_cost=0

		for t in range(self.inputs['n_t']):

			transmission_purchase_cost+=self.inputs['c_tp'][t]*self.model.u_tp[t]
			transmission_sell_cost+=self.inputs['c_ts'][t]*self.model.u_ts[t]

			for s in range(self.inputs['n_s']):

				transmission_purchase_deviation_cost+=(
					self.inputs['p_s'][s]*self.inputs['c_tpd'][t]*self.model.u_tpd[t,s])

				transmission_sell_deviation_cost+=(
					self.inputs['p_s'][s]*self.inputs['c_tsd'][t]*self.model.u_tsd[t,s])

		self.model.objective=pyomo.Objective(
			expr=transmission_purchase_cost+transmission_sell_cost+\
			transmission_purchase_deviation_cost+transmission_sell_deviation_cost)

	def Conservation_of_Energy(self):

		self.model.coe=pyomo.ConstraintList()

		for t in self.model.T:

			for s in self.model.S:

				generation_power=sum(
					self.model.u_g[t,r,s] for r in self.model.R)

				pumping_power=sum(
					self.model.u_p[t,r,s] for r in self.model.R)

				transmission_purchase_power=self.model.u_tp[t]+self.model.u_tpd[t,s]

				transmission_sell_power=self.model.u_ts[t]+self.model.u_tsd[t,s]

				demand_power=self.inputs['y_e'][t,s]

				self.model.coe.add(
					expr=(generation_power-pumping_power+
						transmission_purchase_power-
						transmission_sell_power-
						demand_power==0))

	def Bounds(self):

		self.model.bounds=pyomo.ConstraintList()
		self.model.fs=pyomo.ConstraintList()

		for s in range(self.inputs['n_s']):

			for r in range(self.inputs['n_r']):
				level=self.inputs['is'][r]

				for t in range(self.inputs['n_t']):

					level+=(
						self.model.u_p[t,r,s]-
						self.model.u_g[t,r,s]+
						self.inputs['y_f'][t,r,s])

					self.model.bounds.add(
						(self.inputs['lb'][r],level,self.inputs['ub'][r]))

				self.model.fs.add(expr=level==self.inputs['fs'][r])

	def Unit_Commitment(self):

		self.model.unit_commitment=pyomo.ConstraintList()

		for s in range(self.inputs['n_s']):
			for r in range(self.inputs['n_r']):
				for t in range(self.inputs['n_t']):

					self.model.unit_commitment.add(
						expr=(self.inputs['f_g_min'][r]*self.model.u_gc[t,r,s]-
							self.model.u_g[t,r,s]<=0))
					
					self.model.unit_commitment.add(
						expr=(self.inputs['f_g_max'][r]*self.model.u_gc[t,r,s]-
							self.model.u_g[t,r,s]>=0))

					self.model.unit_commitment.add(
						expr=(self.inputs['f_p_min'][r]*self.model.u_pc[t,r,s]-
							self.model.u_p[t,r,s]<=0))
					
					self.model.unit_commitment.add(
						expr=(self.inputs['f_p_max'][r]*self.model.u_pc[t,r,s]-
							self.model.u_p[t,r,s]>=0))

	def Solution(self):
		'''
		From StackOverflow
		https://stackoverflow.com/questions/67491499/
		how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
		'''
		model_vars=self.model.component_map(ctype=pyomo.Var)

		serieses=[]   # collection to hold the converted "serieses"
		for k in model_vars.keys():   # this is a map of {name:pyo.Var}
			v=model_vars[k]

			# make a pd.Series from each    
			s=pd.Series(v.extract_values(),index=v.extract_values().keys())

			# if the series is multi-indexed we need to unstack it...
			if type(s.index[0])==tuple:# it is multi-indexed
				if len(s.index[0])==3:
					s=s.unstack(level=(1,2))
				elif len(s.index[0])==2:
					s=s.unstack(level=1)
			else:
				s=pd.DataFrame(s) # force transition from Series -> df

			# multi-index the columns
			s.columns=pd.MultiIndex.from_tuples([(k,t) for t in s.columns])

			serieses.append(s)

		self.solution=pd.concat(serieses,axis=1)