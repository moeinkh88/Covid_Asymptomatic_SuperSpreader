# reference for model
# https://doi.org/10.1016/j.chaos.2020.109846
# https://doi.org/10.1016/j.chaos.2021.110652
# https://doi.org/10.3390/axioms10030135

using Optim, StatsBase
using FdeSolver
# using Plots,StatsPlots, StatsPlots.PlotMeasures
using SpecialFunctions
using CSV, DataFrames


# Dataset
dataset_CC = CSV.read("time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="Portugal",45:121] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

#preporcessing (map negative values to zero and remove outliers)
₋Ind=findall(C.<0)
C[₋Ind].=0.0
outlier=findall(C.>1500)
C[outlier]=(C[outlier.-1]+C[outlier.-1])/2

dataset_D = CSV.read("time_series_covid19_deaths_global.csv", DataFrame) # all data of Death
DeathData=dataset_D[dataset_D[!,2].=="Portugal",45:120]
TrueF=(Float64.(Vector(DeathData[1,:])))

## System definition

# parameters from initial fit
β=3.7867885928014893 # Transmission coeﬃcient from infected individuals
l=2.282972432258884 # Relative transmissibility of hospitalized patients
β′=0.9407139791767547 # Transmission coeﬃcient due to super-spreaders
β´´=6.767816564146801 # quantifies transmission coefficient due to asymptomatic
κ=0.11784770556372894 # Rate at which exposed become infectious
ρ₁=0.58 # Rate at which exposed people become infected I
ρ₂=0.001 # Rate at which exposed people become super-spreaders
γₐ=0.15459463331892578# Rate of being hospitalized
γᵢ=0.6976487340139972 # Recovery rate without being hospitalized
γᵣ=0.001310080700085496 # Recovery rate of hospitalized patients
δᵢ=0.0010807082772655262 # Disease induced death rate due to infected class
δₚ=0.45500168006745334 # Disease induced death rate due to super-spreaders
δₕ=0.035917596817983934 # Disease induced death rate due to hospitalized class
δₐ=0.0025773545913321608 #denotes the disease induced death rates due to hospitalized individuals

NN=1083.0516691504504

# Initial conditions
N=10280000/NN # Population Size
S0=N-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
X0=[S0, E0, I0, P0, A0, H0, R0, F0] # initial values

tspan=[1,length(C)] # time span [initial time, final time]

par=[N, β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ] # parameters
# Define model 1: with super spreaders
function SIR1(t, u, par)
    # Model parameters.
	N0, β, l, β′, β´´, κ, ρ₁, ρ₂,γₐ, γᵢ, γᵣ,	δᵢ,	δₚ, δₕ, δₐ=par

    # Current state.
    S, E, I, P, A, H, R, F = u
	N= S+E+I+P+A+H+R+F

# ODE
    dS = - β * I * S/N - l * β * H * S/N - β′* P * S/N # susceptible individuals
    dE = β * I * S/N + l * β * H * S/N + β′ *P* S/N - κ * E # exposed individuals
    dI = κ * ρ₁ * E - (γₐ + γᵢ )*I - δᵢ * I #symptomatic and infectious individuals
    dP = κ* ρ₂ * E - (γₐ + γᵢ)*P - δₚ * P # super-spreaders individuals
    dA = κ *(1 - ρ₁ - ρ₂ )* E# infectious but asymptomatic individuals
	dH = γₐ *(I + P ) - γᵣ *H - δₕ *H # hospitalized individuals
	dR = γᵢ * (I + P ) + γᵣ* H # recovery individuals
	dF = δᵢ * I + δₚ* P + δₕ *H # dead individuals
    return [dS, dE, dI, dP, dA, dH, dR, dF]
end
# Define model 12: with super spreaders + asymptomatic coefficients
function SIR2(t, u, par)
    # Model parameters.
	N0, β, l, β′, β´´, κ, ρ₁, ρ₂,γₐ, γᵢ, γᵣ,	δᵢ,	δₚ, δₕ, δₐ=par

    # Current state.
    S, E, I, P, A, H, R, F = u
	N= S+E+I+P+A+H+R+F

# ODE
    dS = - β * I * S/N - l * β * H * S/N - β′* P * S/N - β´´* P * A/N # susceptible individuals
    dE = β * I * S/N + l * β * H * S/N + β′ *P* S/N + β´´* P * A/N - κ * E # exposed individuals
    dI = κ * ρ₁ * E - (γₐ + γᵢ )*I - δᵢ * I #symptomatic and infectious individuals
    dP = κ* ρ₂ * E - (γₐ + γᵢ)*P - δₚ * P # super-spreaders individuals
    dA = κ *(1 - ρ₁ - ρ₂ )* E - δₐ*A# infectious but asymptomatic individuals
	dH = γₐ *(I + P ) - γᵣ *H - δₕ *H # hospitalized individuals
	dR = γᵢ * (I + P ) + γᵣ* H # recovery individuals
	dF = δᵢ * I + δₚ* P + δₐ*A + δₕ *H # dead individuals
    return [dS, dE, dI, dP, dA, dH, dR, dF]
end

## optimazation of death rates δ for integer order model1

function loss_1(b)# loss function
	p=copy(par)
	p[7]=b[1]
	p[12:14]=b[2:4]
	#initial conditions
	_, x = FDEsolver(SIR1, tspan, X0, ones(8), p, h = .1, nc=4)
    IPH=vec(sum(x[1:10:end,[3,4,6]], dims=2))
	F=cumsum(x[1:10:end,8])
    rmsd([C  TrueF], [IPH  F]) # root-mean-square error
end

p_lo_1=vcat(1e-5*ones(4)) #lower bound
p_up_1=[1-ρ₂,1,1,1] # upper bound
p_vec_1=[.4,.4,.5,.5] #  initial guess
Res1=optimize(loss_1,p_lo_1,p_up_1,p_vec_1,Fminbox(BFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Result=optimize(loss_1,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.99), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			Optim.Options(#outer_iterations = 1,
						  #iterations=1,
						  show_trace=true,
						  show_every=2))
p1=vcat(Optim.minimizer(Res1))
par1=copy(par); par1[12:14]=p1[2:4];par1[7]=p1[1]

## optimazation of death rates δ and β'' for integer order model2

function loss_2(b)# loss function
	p=copy(par)
	p[5]=b[1]
	p[7]=b[2]
	p[12:15]=b[3:6]
	#initial conditions
	_, x = FDEsolver(SIR2, tspan, X0, ones(8), p, h = .1, nc=4)
    IPH=vec(sum(x[1:10:end,[3,4,6]], dims=2))
	F=cumsum(x[1:10:end,8])
    rmsd([C  TrueF], [IPH  F]) # root-mean-square error
end

p_lo_2=vcat(1e-5*ones(6)) #lower bound
p_up_2=vcat(10,1-ρ₂,1,1,1,1) # upper bound
p_vec_2=[9,.4,.4,.5,.4,.001] #  initial guess
Res2=optimize(loss_2,p_lo_2,p_up_2,p_vec_2,Fminbox(BFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
			Optim.Options(#outer_iterations = 1,
						 #iterations=1,
						  show_trace=true,
						  show_every=2))
p2=vcat(Optim.minimizer(Res2))
par2=copy(par); par2[5]=p2[1]; par2[7]=p2[2]; par2[12:15]=p2[3:6]


## optimazation of death rates δ for fractional order model1

# function loss_1f(b)# loss function
# 	p=copy(par)
# 	p[12:14]=b[2:4]
# 	order=b[5]
# 	#initial conditions
# 	_, x = FDEsolver(SIR1, tspan, X0, ones(8)*order, p, h = .1)
#     IPH=vec(sum(x[1:10:end,[3,4,6]], dims=2))
# 	F=x[1:10:end,8]
#     rmsd([C  TrueF], [IPH  F]) # root-mean-square error
# end
#
# p_lo_1f=[1e-5,1e-5,1e-5,1e-5,.5] #lower bound
# p_up_1f=[.999,1,1,1,.999] # upper bound
# p_vec_1f=[.58,.04,.4,.039,.9] #  initial guess
# Res1f=optimize(loss_1f,p_lo_1f,p_up_1f,p_vec_1f,Fminbox(BFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# 			Optim.Options(#outer_iterations = 1,
# 						 # iterations=1,
# 						  show_trace=true,
# 						  show_every=20))
# p1f=vcat(Optim.minimizer(Res1f))
# par1f=copy(par); par1f[7]=p1f[1]; par1f[12:14]=p1f[2:4]; μ1=p1f[5]


## optimazation of death rates δ and β'' for fractional order model2

# function loss_2f(b)# loss function
# 	p=copy(par)
# 	p[5]=b[1]
# 	p[7]=b[2]
# 	p[12:15]=b[3:6]
# 	order= b[7]
# 	#initial conditions
# 	_, x = FDEsolver(SIR2, tspan, X0, ones(8)*order, p, h = .1)
#     IPH=vec(sum(x[1:10:end,[3,4,6]], dims=2))
# 	F=x[1:10:end,8]
#     rmsd([C  TrueF], [IPH  F]) # root-mean-square error
# end
#
# p_lo_2f=[1e-5,1e-5,1e-5,1e-5,1e-5,1e-5,.5] #lower bound
# p_up_2f=[20,.999,1,1,1,1,.9999] # upper bound
# p_vec_2f=[6,.58,.4,.4,.5,.1,.9] #  initial guess
# Res2f=optimize(loss_2f,p_lo_2f,p_up_2f,p_vec_2f,Fminbox(BFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# 			Optim.Options(#outer_iterations = 1,
# 						 # iterations=1,
# 						  show_trace=true,
# 						  show_every=20))
# p2f=vcat(Optim.minimizer(Res2f))
# par2f=copy(par); par2f[5]=p2f[1];par2f[7]=p2f[2]; par2f[12:15]=p2f[3:6]; μ2=p2f[7]

## optimazation of parameters and order of incommensurate fractional order model1
function loss_1f8(b)
	p=copy(par)
	p[7]=b[1]
	p[12:14]=b[2:4]
	order = b[5:12] # order of derivatives
	if size(X0,2) != Int64(ceil(maximum(order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, order)
		order[indx]=ones(length(indx))
	end
	_, x = FDEsolver(SIR1, tspan, X0, order, p, h = .1,nc=4)
    IPH=vec(sum(x[1:10:end,[3,4,6]], dims=2))
	F=x[1:10:end,8]
    rmsd([C  TrueF], [IPH  F]) # root-mean-square error
end

p_lo=vcat(1e-5*ones(4),.7*ones(8))
p_up=vcat(1-ρ₂,ones(11))
pvec=vcat(.4,.4,.4,.4,ones(8)*.9)
Res1F8=optimize(loss_1f8,p_lo,p_up,pvec,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
			Optim.Options(outer_iterations = 8,
						 iterations=200,
						  show_trace=true,
						  show_every=20))
p1f8=vcat(Optim.minimizer(Res1F8))
par1f8=copy(par); par1f8[7]=p1f8[1];par1f8[12:14]=p1f8[2:4]; μ1f8=p1f8[5:12]
#
## optimazation of parameters and order of incommensurate fractional order model2

function loss_2f8(b)# loss function
	p=copy(par)
	p[5]=b[1]
	p[7]=b[2]
	p[12:15]=b[3:6]
	order = b[7:14]# order of derivatives
	if size(X0,2) != Int64(ceil(maximum(order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, order)
		order[indx]=ones(length(indx))
	end
	#initial conditions
	_, x = FDEsolver(SIR2, tspan, X0, order, p, h = .1, nc=4)
    IPH=vec(sum(x[1:10:end,[3,4,6]], dims=2))
	F=x[1:10:end,8]
    rmsd([C  TrueF], [IPH  F]) # root-mean-square error
end

p_lo=vcat(1e-5*ones(6),.7*ones(8))
p_up=vcat(10,1-ρ₂,1,1,1,1,ones(8))
pvec=vcat(6,.5,.4,.4,.4,.1,ones(8)*.9)
Res2F8=optimize(loss_2f8,p_lo,p_up,pvec,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
			Optim.Options(outer_iterations = 8,
						 iterations=200,
						  show_trace=true,
						  show_every=20))
p2f8=vcat(Optim.minimizer(Res2F8))
par2f8=copy(par); par2f8[5]=p2f8[1]; par2f8[7]=p2f8[2]; par2f8[12:15]=p2f8[3:6]; μ2f8=p2f8[7:14]


## results

_, x1 = FDEsolver(SIR1, tspan, X0, ones(8), par1, h = .1) # solve ode model
# _, x1f = FDEsolver(SIR1, tspan, X0, μ1*ones(8), par1f, h = .1) # solve commensurate fode model
_, x1f8 = FDEsolver(SIR1, tspan, X0, μ1f8, par1f8, h = .1) # solve incommensurate fode model
_, x2 = FDEsolver(SIR2, tspan, X0, ones(8), par2, h = .1)
# _, x2f = FDEsolver(SIR2, tspan, X0, μ2*ones(8), par2f, h = .1)
_, x2f8 = FDEsolver(SIR2, tspan, X0, μ2f8, par2f8, h = .1) # solve incommensurate fode model

IPH1=sum(x1[1:10:end,[3,4,6]], dims=2);F1=x1[1:10:end,8]
# IPH1f=sum(x1f[1:10:end,[3,4,6]], dims=2);F1f=x1f[1:10:end,8]
IPH1f8=sum(x1f8[1:10:end,[3,4,6]], dims=2);F1f8=x1f8[1:10:end,8]
IPH2=sum(x2[1:10:end,[3,4,6]], dims=2);F2=x2[1:10:end,8]
# IPH2f=sum(x2f[1:10:end,[3,4,6]], dims=2);F2f=x2f[1:10:end,8]
IPH2f8=sum(x2f8[1:10:end,[3,4,6]], dims=2);F2f8=x2f8[1:10:end,8]

Err1=rmsd([C  TrueF], [IPH1  F1]) # RMSE for ode model
# Err1f=rmsd([C  TrueF], [IPH1f  F1f]) # RMSE for commensurate fode model
Err1f8=rmsd([C  TrueF], [IPH1f8  F1f8]) # RMSE for incommensurate fode model
Err2=rmsd([C  TrueF], [IPH2  F2]) # RMSE for ode model
# Err2f=rmsd([C  TrueF], [IPH2f  F2f]) # RMSE for commensurate fode model
Err2f8=rmsd([C  TrueF], [IPH2f8  F2f8]) # RMSE for incommensurate fode model

display(["Err1=",Err1,"par1=",par1,
		# "Err1f=",Err1f,"par1f=",par1f,"order1f=",μ1,
		"Err1f8=",Err1f8,"par1f8=",par1f8,"order1f8=",μ1f8,
		"Err2=",Err2,"par2=",par2,
		# "Err2f=",Err2f,"par2f=",par2f,"order2f=",μ2,
		"Err2f8=",Err2f8,"par2f8=",par2f8,"order2f8=",μ2f8])
