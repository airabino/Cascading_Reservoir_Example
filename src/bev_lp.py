import sys
import time
import argparse
import numpy as np
import numpy.random as rand
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from pulp import *
from copy import copy,deepcopy

from datetime import datetime
from itertools import chain
from scipy.stats import binom

from .utilities import IsIterable,FullFact

class Vehicle():

	def __init__(self,
		itinerary,
		destination_charger_likelihood=.1,
		consumption=478.8,
		battery_capacity=82*3.6e6,
		initial_soc=.5,
		final_soc=.5,
		payment_penalty=60,
		destination_charger_power=12100,
		en_route_charger_power=150000,
		home_charger_power=12100,
		work_charger_power=12100,
		ac_dc_conversion_efficiency=.88,
		max_soc=1,
		min_range=25000,
		quanta_soc=50,
		quanta_ac_charging=2,
		quanta_dc_charging=10,
		max_en_route_charging=7200,
		travel_penalty=15*60,
		final_soc_penalty=1e10,
		bounds_violation_penalty=1e50,
		tiles=7,
		time_multiplier=1,
		cost_multiplier=60,
		electricity_times=np.arange(0,25,1)*3600,
		electricity_rates=np.array([
			0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,0.055,
			0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,0.075,
			.123,.123,.123,.123,0.055,0.055,0.055
			])/3.6e6*10,
		low_rate_charging_multiplier=1.1, #[-]
		high_rate_charging_multiplier=1.5, #[-]
		rng_seed=0
		):

		self.initial_soc=initial_soc #[-]
		self.final_soc=final_soc #[-]
		self.payment_penalty=payment_penalty #[-]
		self.destination_charger_likelihood=destination_charger_likelihood #[-]
		self.destination_charger_power=destination_charger_power #[W
		self.en_route_charger_power=en_route_charger_power #[W]
		self.home_charger_power=home_charger_power #[W]
		self.work_charger_power=work_charger_power #[W]
		self.consumption=consumption #[J/m]
		self.battery_capacity=battery_capacity #[J]
		self.ac_dc_conversion_efficiency=ac_dc_conversion_efficiency #[-]
		self.max_soc=max_soc #[-]
		self.min_range=min_range #[m]
		self.min_soc=self.min_range*self.consumption/self.battery_capacity #[-]
		self.quanta_soc=quanta_soc #[-]
		self.x=np.linspace(0,1,quanta_soc) #[-]
		self.quanta_ac_charging=quanta_ac_charging #[-]
		self.quanta_dc_charging=quanta_dc_charging #[-]
		self.u1=np.linspace(0,1,quanta_ac_charging) #[s]
		self.max_en_route_charging=max_en_route_charging #[s]
		self.u2=np.linspace(0,1,quanta_dc_charging) #[s]
		self.travel_penalty=travel_penalty #[s]
		self.final_soc_penalty=final_soc_penalty
		self.bounds_violation_penalty=bounds_violation_penalty
		self.tiles=tiles #[-]
		self.time_multiplier=time_multiplier/(time_multiplier+cost_multiplier) #[-]
		self.cost_multiplier=cost_multiplier/(time_multiplier+cost_multiplier) #[-]
		self.electricity_times=electricity_times #[-]
		self.electricity_rates=electricity_rates #[$/J]
		self.low_rate_charging_multiplier=low_rate_charging_multiplier #[-]
		self.high_rate_charging_multiplier=high_rate_charging_multiplier #[-]
		self.rng_seed=rng_seed
		
		self.itineraryArrays(itinerary)

	def itineraryArrays(self,itinerary):

		person=itinerary['PERSONID'].copy().to_numpy()
		itinerary=itinerary[person==person.min()]

		#Adding trip and dwell durations
		durations=itinerary['TRVLCMIN'].copy().to_numpy()
		dwell_times=itinerary['DWELTIME'].copy().to_numpy()


		#Fixing any non-real dwells
		dwell_times[dwell_times<0]=dwell_times[dwell_times>=0].mean()
		
		#Padding with overnight dwell
		sum_of_times=durations.sum()+dwell_times[:-1].sum()

		if sum_of_times>=1440:
			ratio=1440/sum_of_times
			dwell_times*=ratio
			durations*=ratio
		else:
			final_dwell=1440-durations.sum()-dwell_times[:-1].sum()
			dwell_times[-1]=final_dwell

		#Populating itinerary arrays
		self.trip_distances=np.tile(
		itinerary['TRPMILES'].to_numpy(),self.tiles)*1609.34 #[m]
		print(itinerary['TRPMILES'].sum())
		self.trip_times=np.tile(durations,self.tiles)*60 #[s]
		self.trip_mean_speeds=self.trip_distances/self.trip_times #[m/s]
		self.dwells=np.tile(dwell_times,self.tiles)*60
		self.location_types=np.tile(itinerary['WHYTRP1S'].to_numpy(),self.tiles)
		self.is_home=self.location_types==1
		self.is_work=self.location_types==10
		self.is_other=(~self.is_home&~self.is_work)

		self.destination_charger_power_array=np.array(
			[self.destination_charger_power]*len(self.dwells))
		if self.rng_seed:
			seed=self.rng_seed
		else:
			seed=np.random.randint(1e6)

		generator=np.random.default_rng(seed=seed)
		charger_selection=generator.random(len(self.destination_charger_power_array))
		no_charger=charger_selection>=self.destination_charger_likelihood

		self.destination_charger_power_array[no_charger]=0

		#Adding home chargers to home destinations
		self.destination_charger_power_array[self.is_home]=self.home_charger_power

		#Adding work chargers to work destinations
		self.destination_charger_power_array[self.is_work]=self.work_charger_power

		# print(self.destination_charger_power_array)

		#Cost of charging
		trip_start_times=itinerary['STRTTIME'].to_numpy()*60
		trip_end_times=itinerary['ENDTIME'].to_numpy()*60
		trip_mean_times=(trip_start_times+trip_end_times)/2
		dwell_start_times=itinerary['ENDTIME'].to_numpy()*60
		dwell_end_times=itinerary['ENDTIME'].to_numpy()*60+dwell_times*60
		dwell_mean_times=(dwell_start_times+dwell_end_times)/2

		self.en_route_charge_cost_array=np.tile(
			self.high_rate_charging_multiplier*np.interp(
			trip_mean_times,self.electricity_times,self.electricity_rates),self.tiles)
		self.destination_charge_cost_array=np.tile(
			self.low_rate_charging_multiplier*np.interp(
			dwell_mean_times,self.electricity_times,self.electricity_rates),self.tiles)

		self.discharge_events=self.trip_distances*self.consumption

	def Copy(self):

		return deepcopy(self)

'''
Produce a factorial set of Vehicle objects
'''
def VehicleFactorial(vehicle,p,n):

	n_home_dwells=vehicle.is_home.sum()

	design=FullFact([2 for num in range(n_home_dwells)]).astype(int)
	n_available=design.sum(axis=1)

	likelihood=binom.pmf(n_available,n_home_dwells*np.ones(len(design)),p)
	selection_indices=np.array(
		[np.round(num/(n-1)*(len(design)-1)) for num in range(n)]).astype(int)

	selection_indices=likelihood>=.1
	design=design[selection_indices]
	likelihood=likelihood[selection_indices]

	vehicles=[]
	for idx in range(n):

		vehicle_temp=vehicle.Copy()
		vehicle_temp.destination_charger_power_array[vehicle_temp.is_home]*=design[idx]

		vehicles.append(vehicle_temp)

	return vehicles,likelihood

'''
Solve the charging optimization MILP
'''
def LinearProblem(vehicle,handle='',write=False):

	if not handle:
		now=datetime.now()
		handle='LP_'+now.strftime("%m%d%Y_%H%M%S")

	#Initialize the problem
	problem=LpProblem(handle,LpMinimize)

	acced,dcced,acceb,dcceb=VariableGeneration(vehicle)

	objective=ObjectiveFunction(vehicle,acced,dcced,acceb,dcceb)

	problem+=LpAffineExpression(objective)

	#Event commitment constraints
	problem=EventCommitmentConstraints(problem,vehicle,acced,dcced,acceb,dcceb)

	#Computing SOC constraint values
	problem=SOCConstraints(problem,vehicle,acced,dcced)

	if write:
		problem.writeLP(problem.name)

	return problem

'''
Solve individual MILPs for constituents to S-MILP
'''
def SolveConstituents(vehicles,solver,likelihood=[],handle='',write=False):

	if not likelihood:
		likelihood=np.ones(len(vehicles))

	sic=np.zeros(len(vehicles))
	for idx,vehicle in enumerate(vehicles):

		problem=LinearProblem(vehicle,handle=handle,write=write)

		res=problem.solve(solver)
		solution=Solution(problem)
		# print(solution)

		sic[idx]=ComputeSIC(vehicle,solution)

	print(sic)
	sic*=likelihood

	return sic.sum()/likelihood.sum()

'''
Solve the charging optimization S-MILP
'''
def StochasticLinearProblem(vehicles,handle='',write=False):

	if not handle:
		now=datetime.now()
		handle='LP_'+now.strftime("%m%d%Y_%H%M%S")

	#Initialize the problem
	problem=LpProblem(handle,LpMinimize)

	acced,dcced,acceb,dcceb=VariableGeneration(vehicles[0])

	objective=ObjectiveFunction(vehicles[0],acced,dcced,acceb,dcceb)

	problem+=LpAffineExpression(objective)

	#Event commitment constraints
	for idx,vehicle in enumerate(vehicles):
		problem=EventCommitmentConstraints(problem,vehicle,acced,dcced,acceb,dcceb,j=idx)

	#Computing SOC constraint values
	for idx,vehicle in enumerate(vehicles):
		problem=SOCConstraints(problem,vehicle,acced,dcced,j=idx)

	if write:
		problem.writeLP(problem.name)

	return problem

'''
Generate charging optimization MILP variables
'''
def VariableGeneration(vehicle):

	n_t=len(vehicle.dwells)

	#Creating the variables - low and high rate charging
	acced=Variable1D(n_t,'acced',np.vstack((np.zeros(n_t),vehicle.dwells)).T)
	acceb=Variable1D(n_t,'acceb',[0,1],cat='Binary')

	dcced=Variable1D(n_t,'dcced',np.vstack((np.zeros(n_t),
		np.ones(n_t)*vehicle.max_en_route_charging)).T)
	dcceb=Variable1D(n_t,'dcceb',[0,1],cat='Binary')

	return acced,dcced,acceb,dcceb

'''
Generate charging optimization MILP objective function
'''
def ObjectiveFunction(vehicle,acced,dcced,acceb,dcceb):

	n_t=len(vehicle.dwells)

	#Payment penalty for AC public charge events
	ac_charge_event_penalty=Expression1D(n_t,acceb,
		vehicle.is_other.astype(int)*vehicle.payment_penalty)

	#Travel and payment penalty for DC public charge events
	dc_charge_event_penalty=Expression1D(n_t,dcceb,
		np.ones(n_t)*(vehicle.payment_penalty+vehicle.travel_penalty))

	#Charging time penalty for DC public charging events
	dc_charge_time_penalty=Expression1D(n_t,dcced,
		np.ones(n_t))

	objective=[*list(itertools.chain(ac_charge_event_penalty)),\
		*list(itertools.chain(dc_charge_event_penalty)),\
		*list(itertools.chain(dc_charge_time_penalty))]

	return objective

'''
Generate charging optimization MILP event commitment (unit commitment) constraints
'''
def EventCommitmentConstraints(problem,vehicle,acced,dcced,acceb,dcceb,j=0):

	n_t=len(vehicle.dwells)

	for idx in range(n_t):

		lhs=LpAffineExpression(((acceb[idx],1),(acced[idx],-1)))
		problem+=LpConstraint(lhs,LpConstraintLE,
			name=f'ac_unit_commitment_lower_{idx}_{j}',rhs=0)

		lhs=LpAffineExpression(((acceb[idx],vehicle.dwells[idx]),(acced[idx],-1)))
		problem+=LpConstraint(lhs,LpConstraintGE,
			name=f'ac_unit_commitment_upper_{idx}_{j}',rhs=0)

		lhs=LpAffineExpression(((dcceb[idx],1),(dcced[idx],-1)))
		problem+=LpConstraint(lhs,LpConstraintLE,
			name=f'dc_unit_commitment_lower_{idx}_{j}',rhs=0)

		lhs=LpAffineExpression(((dcceb[idx],vehicle.max_en_route_charging),(dcced[idx],-1)))
		problem+=LpConstraint(lhs,LpConstraintGE,
			name=f'dc_unit_commitment_upper_{idx}_{j}',rhs=0)

	return problem

'''
Generate charging optimization MILP SOC constraints
'''
def SOCConstraints(problem,vehicle,acced,dcced,j=0):

	n_t=len(vehicle.dwells)

	cumulative_discharge=np.cumsum(vehicle.discharge_events)
	initial_energy=vehicle.initial_soc*vehicle.battery_capacity
	diff_final_energy=vehicle.final_soc*vehicle.battery_capacity-initial_energy

	battery_state=[]
	for idx in range(n_t):
		battery_state.append((acced[idx],vehicle.destination_charger_power_array[idx]))
		battery_state.append((dcced[idx],vehicle.en_route_charger_power))

		if idx==n_t-1:
			# print(battery_state)
			problem+=LpConstraint(
				LpAffineExpression(
					battery_state,
					# constant=0,
					constant=-cumulative_discharge[idx]
				),
				LpConstraintGE,
				name=f'battery_state_final_{j}',
				rhs=diff_final_energy)

		else:
			problem+=LpConstraint(
				LpAffineExpression(
					battery_state,
					constant=-cumulative_discharge[idx]
					# constant=0
					),
				LpConstraintGE,
				name=f'battery_state_lb_{idx}_{j}',
				rhs=(vehicle.min_soc-vehicle.initial_soc)*vehicle.battery_capacity)

			problem+=LpConstraint(
				LpAffineExpression(
					battery_state,
					constant=-cumulative_discharge[idx],
					# constant=0
					),
				LpConstraintLE,
				name=f'battery_state_ub_{idx}_{j}',
				rhs=(vehicle.max_soc-vehicle.initial_soc)*vehicle.battery_capacity)

	return problem

'''
Compute Inconvenience Score (S_IC) for a given solution to charging optimization problem
'''
def ComputeSIC(vehicle,solution):

	dedicated_time=(
		solution['acceb']*vehicle.is_other.astype(int)*vehicle.payment_penalty+
		solution['dcceb']*(vehicle.payment_penalty+vehicle.travel_penalty)+
		solution['dcced']).sum()/60

	trip_distance=vehicle.trip_distances.sum()/1000

	return dedicated_time/trip_distance

'''
Extract solution variables into dict
'''
def SolutionDict(problem):
	return {v.name:v.varValue for v in problem.variables()}

'''
Preparing solution variables into output format
'''
def Solution(problem):

	solution_dict=SolutionDict(problem)

	keys=list(solution_dict.keys())
	variables={}

	for idx,key in enumerate(keys):
		
		elements=key.split('_')

		if elements[0] in variables.keys():

			variables[elements[0]]['indices'].append([int(e) for e in elements[1:]])
			variables[elements[0]]['values'].append(solution_dict[key])

		else:

			variables[elements[0]]={'indices':[[int(e) for e in elements[1:]]],
				'values':[solution_dict[key]]}

	solution={}

	for key in variables.keys():

		indices=variables[key]['indices']
		values=variables[key]['values']

		max_idx=np.array(indices).max(axis=0)+1

		out_data=np.empty(tuple(max_idx))

		for idx,index in enumerate(indices):

			out_data[*index]=values[idx]

		solution[key]=out_data

	return solution

'''
Create 1D ndarray containing LpVariable
'''
def Variable1D(n,tag,bounds,cat='Continuous'):
	if len(bounds)<n:
		bounds=np.tile(bounds,(n,1))
	u=np.empty((n,),dtype=object)
	for idx in range(n):
		u[idx]=LpVariable(f'{tag}_{idx}',*bounds[idx],cat)

	return u

'''
Create 1D ndarray containing LpVariable
'''
def BinaryVariable1D(n,tag,cat='Binary'):

	u=np.empty((n,),dtype=object)
	for idx in range(n):
		u[idx]=LpVariable(f'{tag}_{idx}',cat)

	return u

'''
Create 1D ndarray containing tuples for LpAffineExpression for 1D variable
'''
def Expression1D(n,u,multiplier):
	if not IsIterable(multiplier):
		multiplier=np.tile(multiplier,(n,1))

	expression=np.empty((n,),dtype=object)

	for idx in range(n):
		expression[idx]=(u[idx],multiplier[idx])

	return expression