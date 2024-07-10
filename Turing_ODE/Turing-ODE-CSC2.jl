# reference for model
# https://doi.org/10.1016/j.chaos.2020.109846
# https://doi.org/10.1016/j.chaos.2021.110652
# https://doi.org/10.3390/axioms10030135

using Optim, StatsBase
using DifferentialEquations, Turing, LinearAlgebra
# using Plots,StatsPlots, StatsPlots.PlotMeasures
using SpecialFunctions
using CSV, HTTP, DataFrames, Dates


# Dataset
dataset_CC = CSV.read("time_series_covid19_confirmed_global.csv", DataFrame) # all data of confirmed
Confirmed=dataset_CC[dataset_CC[!,2].=="India",45:121] #comulative confirmed data of Portugal from 3/2/20 to 5/17/20
C=diff(Float64.(Vector(Confirmed[1,:])))# Daily new confirmed cases

#preporcessing (map negative values to zero and remove outliers)
₋Ind=findall(C.<0)
C[₋Ind].=0.0
outlier=findall(C.>1500)
C[outlier]=(C[outlier.-1]+C[outlier.-1])/2


dataset_D = CSV.read("time_series_covid19_deaths_global.csv", DataFrame) # all data of Death
DeathData=dataset_D[dataset_D[!,2].=="Portugal",45:120]
TrueF=(Float64.(Vector(DeathData[1,:])))

dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="Portugal",45:120]
TrueR=(Float64.(Vector(RData[1,:])))

## System definition

# parameters
β=2.55 # Transmission coeﬃcient from infected individuals
l=1.56 # Relative transmissibility of hospitalized patients
β′=7.65 # Transmission coeﬃcient due to super-spreaders
β´´=0 # quantifies transmission coefficient due to asymptomatic
κ=0.25 # Rate at which exposed become infectious
ρ₁=0.58 # Rate at which exposed people become infected I
ρ₂=0.001 # Rate at which exposed people become super-spreaders
γₐ=0.94 # Rate of being hospitalized
γᵢ=0.27 # Recovery rate without being hospitalized
γᵣ=0.5 # Recovery rate of hospitalized patients
δᵢ=1/23 # Disease induced death rate due to infected class
δₚ=1/23 # Disease induced death rate due to super-spreaders
δₕ=1/23 # Disease induced death rate due to hospitalized class
δₐ=0 #denotes the disease induced death rates due to hospitalized individuals


# Initial conditions
E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
N=10280000/875 # Population Size
tspan=[1,length(C)] # time span [initial time, final time]

par=[N, β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ] # parameters
# Define SIR model
function SIR(dx, x, par, t)
    # Model parameters.
	N, β, l, β′, β´´, κ, ρ₁, ρ₂,γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ=par

    # Current state.
    S, E, I, P, A, H, R, F = x

# ODE
    dx[1] = - β * I * S/N - l * β * H * S/N - β′* P * S/N - β´´* P * A/N # susceptible individuals
    dx[2] = β * I * S/N + l * β * H * S/N + β′ *P* S/N + β´´* P * A/N - κ * E # exposed individuals
    dx[3] = κ * ρ₁ * E - (γₐ + γᵢ )*I - δᵢ * I #symptomatic and infectious individuals
    dx[4] = κ* ρ₂ * E - (γₐ + γᵢ)*P - δₚ * P # super-spreaders individuals
    dx[5] = κ *(1 - ρ₁ - ρ₂ )* E - δₐ*A# infectious but asymptomatic individuals
	dx[6] = γₐ *(I + P ) - γᵣ *H - δₕ *H # hospitalized individuals
	dx[7] = γᵢ * (I + P ) + γᵣ* H # recovery individuals
	dx[8] = δᵢ * I + δₚ* P + δₐ*A + δₕ *H # dead individuals
    return nothing
end

## optimazation of β for integer order model
X0=[N-5, E0, I0, P0, A0, H0, R0, F0] # initial values
prob = ODEProblem(SIR, X0, tspan, par)

@model function fitprob(data, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    β ~ truncated(Normal(1, 5); lower=1, upper=5)
    NN ~ truncated(Normal(1, 1000); lower=1, upper=10000)
	κ ~ truncated(Normal(0, 4); lower=0, upper=4)
	l ~ truncated(Normal(0, 20); lower=0, upper=20)
	β′ ~ truncated(Normal(0, 20); lower=0, upper=20)
	γₐ ~ truncated(Normal(0,1); lower=0, upper=1)
 	γᵢ~ truncated(Normal(0,1); lower=0, upper=1)
	γᵣ~ truncated(Normal(0,1); lower=0, upper=1)
 	δᵢ~ truncated(Normal(0,1); lower=0, upper=1)
	δₚ~ truncated(Normal(0,1); lower=0, upper=1)
	δₕ~ truncated(Normal(0,1); lower=0, upper=1)

	p=[10280000/NN, β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ]
	S0=10280000/NN-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
	X0=[S0, E0, I0, P0, A0, H0, R0, F0] # initial values

	prob = remake(prob; p = p, u0 = X0)
	x = solve(prob, alg_hints=[:stiff]; saveat=1)
	IPH=x[3,:] .+ x[4,:] .+ x[6,:]
	R=x[7,:]
	F=x[8,:]
	pred=[IPH R F]

	# Observations
	for i in 1:length(pred[1,:])
	data[:,i] ~ MvNormal(pred[:,i], σ^2 * I)
	end
	# for i in 1:length(appX)
	# data[i, :] ~ MvNormal(appX[i], σ^2)
	# end

    return nothing
end

model = fitprob([C TrueR TrueF], prob)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
nChain=5000
chain = sample(model, NUTS(0.65), MCMCSerial(), nChain, 3; progress=false)

posterior_samples = sample(chain[[:β, :NN, :κ,:l, :β′, :γₐ, :γᵢ, :γᵣ,:δᵢ ,:δₚ , :δₕ]], nChain; replace=false)

Err=zeros(nChain)
for i in 1:nChain
	pp=Array(posterior_samples.value[:,:,1])[i,:]
	β, NN, κ, l, β′, γₐ,γᵢ,γᵣ,δᵢ,δₚ , δₕ = pp[1:11]
	p = [10280000/NN, β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ]
	S0=10280000/NN-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
	X0=[S0, E0, I0, P0, A0, H0, R0, F0]

	prob = ODEProblem(SIR, X0, tspan, p)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	x=hcat(sol.u...)
	S=x[1,:]; E=x[2,:]; I1=x[3,:]; P=x[4,:];
	A=x[5,:]; H=x[6,:]; R=x[7,:]; F=x[8,:]
    appX=vec(sum(I1 .+ P .+ H, dims=2))
	predict=hcat(appX, R, F)

	Err[i]=rmsd(hcat(C, TrueR, TrueF), predict)

end

valErr,indErr=findmin(Err)

display([valErr, indErr])
display(Array(posterior_samples.value[:,:,1])[indErr,:])
display(chain)
