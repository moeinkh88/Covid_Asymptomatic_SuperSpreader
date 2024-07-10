# reference for model
# https://doi.org/10.1016/j.chaos.2020.109846
# https://doi.org/10.1016/j.chaos.2021.110652
# https://doi.org/10.3390/axioms10030135

using Optim, StatsBase
using DifferentialEquations, Turing, LinearAlgebra
using Plots,StatsPlots, StatsPlots.PlotMeasures
using SpecialFunctions
using CSV, HTTP, DataFrames, Dates


# Dataset
repo=HTTP.get("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv") # dataset of Covid from CSSE
dataset_CC = CSV.read(repo.body, DataFrame) # all data of confirmed
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
    dx[5] = abs(κ *(1 - ρ₁ - ρ₂ )* E - δₐ*A) # infectious but asymptomatic individuals
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

	p=[10280000/NN, β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ]
	S0=10280000/NN-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
	X0=[S0, E0, I0, P0, A0, H0, R0, F0] # initial values

	prob = remake(prob; p = p, u0 = X0)
	xx = solve(prob, alg_hints=[:stiff]; saveat=1)
	x=hcat(xx.u...)
	S=x[1,:]; E=x[2,:]; I=x[3,:]; P=x[4,:];
	A=x[5,:]; H=x[6,:]; R=x[7,:]; F=x[8,:]
    appX=vec(sum(I .+ P .+ H, dims=2))
	predict=[appX, R, F]

	# Observations
	for i in 1:length(predict)
	data[i , :] ~ MvNormal(predict[i], σ^2 * I)
	end

    return nothing
end

model = fitprob([C, TrueR, TrueF],prob)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
chain = sample(model, NUTS(0.65), MCMCSerial(), 2000, 3; progress=false)

posterior_samples = sample(chain[[:β,:NN]], 2000; replace=false)

Err=zeros(2000)
for i in 1:2000
	pp=Array(posterior_samples.value[:,:,1])[i,:]
	β, NN = pp[1:2]
	p = [10280000/NN, β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ]
	S0=10280000/NN-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
	X0=[S0, E0, I0, P0, A0, H0, R0, F0]

	prob = ODEProblem(SIR, X0, tspan, p)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	x=hcat(sol.u...)
	S=x[1,:]; E=x[2,:]; I=x[3,:]; P=x[4,:];
	A=x[5,:]; H=x[6,:]; R=x[7,:]; F=[8, :]
    appX=vec(sum(I .+ P .+ H, dims=2))
	predict=[appX, R, F]

	Err[i]=rmsd(C, predict)

end

valErr,indErr=findmin(Err)
Result=Array(posterior_samples.value[:,:,1])[indErr,:]

pp=Array(posterior_samples.value[:,:,1])[indErr,:]
β, NN = pp[1:2]
β, NN, l, β′, κ, ρ₁, ρ₂, δᵢ, δₚ, δₕ = [ 1.0429235553133118, 682.1432232114977,   2.823423729571919,   8.608114128833526,   0.04305944594015574,   0.9860688602224335,   0.25046852548014764,   0.17757268303442156,   0.2005382987001781,   0.6064056472709013]
p = [10280000/NN, β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ]
S0=10280000/NN-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
X0=[S0, E0, I0, P0, A0, H0, R0, F0]

prob = ODEProblem(SIR, X0, tspan, p)
sol = solve(prob, alg_hints=[:stiff]; saveat=1)

x=hcat(sol.u...)
S=x[1,:]; E=x[2,:]; I1=x[3,:]; P=x[4,:];
A=x[5,:]; H=x[6,:]; R=x[7,:]; F1=x[8,:];
plot(sol.t, I1 .+ P .+ H)
scatter!(C)
plot(sol.t, F1)
scatter!(TrueF)
plot(sol.t, R)
scatter!(TrueR)
##
@model function fitprob2(data, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    β ~ truncated(Normal(.1, 5); lower=.1, upper=5)
	β´´ ~ truncated(Normal(0, 100); lower=0, upper=100)
	δₐ ~ truncated(Normal(0, 100); lower=0, upper=100)
    NN ~ truncated(Normal(1, 1000); lower=1, upper=10000)

	p=[10280000/NN, β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ]
	S0=10280000/NN-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
	X0=[S0, E0, I0, P0, A0, H0, R0, F0] # initial values

	prob = remake(prob; p = p, u0 = X0)
	xx = solve(prob, alg_hints=[:stiff]; saveat=1)
	x=hcat(xx.u...)
	S=x[1,:]; E=x[2,:]; I=x[3,:]; P=x[4,:];
	A=x[5,:]; H=x[6,:]; R=x[7,:]; F=x[8,:]
    appX=vec(sum(I .+ P .+ H, dims=2))

	# Observations
	for i in 1:length(appX)
	data[i, :] ~ MvNormal(appX[i], σ^2 * I)
	end

    return nothing
end

model = fitprob2(C,prob)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
chain = sample(model, NUTS(0.65), MCMCSerial(), 3000, 3; progress=false)

posterior_samples = sample(chain[[:β,:β´´,:δₐ,:NN]], 3000; replace=false)

Err=zeros(3000)
for i in 1:3000
	pp=Array(posterior_samples.value[:,:,1])[i,:]
	β, β´´, δₐ, NN = pp[1:4]
	p = [10280000/NN, β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ]
	S0=10280000/NN-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
	X0=[S0, E0, I0, P0, A0, H0, R0, F0]

	prob = ODEProblem(SIR, X0, tspan, p)
	sol = solve(prob, alg_hints=[:stiff]; saveat=1)
	x=hcat(sol.u...)
	S=x[1,:]; E=x[2,:]; I=x[3,:]; P=x[4,:];
	A=x[5,:]; H=x[6,:]; R=x[7,:];
    appX=vec(sum(I .+ P .+ H, dims=2))

	Err[i]=rmsd(C, appX)

end

valErr,indErr=findmin(Err)
Result=Array(posterior_samples.value[:,:,1])[indErr,:]

## plotting
DateTick=Date(2020,3,3):Day(1):Date(2020,5,17)
DateTick2= Dates.format.(DateTick, "d u")

t1, x1 = FDEsolver(SIR, tspan, X01, ones(8), par1, h = .1) # solve ode model
_, xf1 = FDEsolver(SIR, tspan, X0, μ1*ones(8), parf1, h = .1) # solve commensurate fode model
_, x2 = FDEsolver(SIR, tspan, X02, ones(8), par2, h = .01, nc=4)
_, xf2 = FDEsolver(SIR, tspan, X0, μ2*ones(8), parf2, h = .1)

X1=sum(x1[1:10:end,[3,4,6]], dims=2)
Xf1=sum(xf1[1:10:end,[3,4,6]], dims=2)
X2=sum(x2[1:100:end,[3,4,6]], dims=2)
Xf2=sum(xf2[1:10:end,[3,4,6]], dims=2)

Err1=rmsd(C, vec(X1)) # RMSE for ode model
Errf1=rmsd(C, vec(Xf1)) # RMSE for commensurate fode model
Err2=rmsd(C, vec(X2)) # RMSE for ode model
Errf2=rmsd(C, vec(Xf2)) # RMSE for commensurate fode model

plot(DateTick2,X1, ylabel="Daily new confirmed cases",
	     label="M1",xrotation=rad2deg(pi/3), linestyle=:dashdot)
plot!(X2,  label="M2", linestyle=:dash)
plot!(Xf1, label="Mf1")
		plot!(Xf2, label="Mf2")
scatter!(C, label= "Real data",legendposition=(.88,.6),legendfontsize=7,
		title = "(b) Portugal" , titleloc = :left, titlefont = font(10))
plPortugal=bar!(["M1" "M2" "Mf1" "Mf2"],[Err1 Err2 Errf1 Errf2], ylabel="RMSD",
				legend=:false, bar_width=2,yguidefontsize=8,xtickfontsize=7,
	    inset = (bbox(0.04, 0.08, 70px, 60px, :right)),
	    subplot = 2,
	    bg_inside = nothing)


savefig(plPortugal,"plPortugal.png")
