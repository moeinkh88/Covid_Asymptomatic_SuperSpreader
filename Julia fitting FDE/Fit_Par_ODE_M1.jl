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
pp=[2.9763470555591796
4.9724275315175905
2.77222207964141
1362.9208020440333
0.03658959949393572
1.605375275205752
0.1375531965076037
0.0560295261424954
0.9179720534039433
0.044242996045119866
0.001724702577525102
0.003473250106648919
0.02042169878603325
0.25088774743033504
0.5272699408450209]
β, β′, β´´, NN, κ, l, γₐ, γᵢ, γᵣ, δᵢ, δₚ, δₕ, δₐ, ρ₂, ρ₁ = pp[1:15]
β′=0
    ρ₂ =0 
    δₚ = 0
par=[10280000/NN,  β, l, β′, β´´, κ, ρ₁, ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ] # parameters

# Initial conditions
N=10280000/NN # Population Size
S0=N-4; E0=0; I0=4; P0=0; A0=0; H0=0; R0=0; F0=0
X0=[N-4, E0, I0, P0, A0, H0, R0, F0] # initial values

tspan=[1,length(C)] # time span [initial time, final time]

# Define model 1: with asymptomatic coefficients (no super spreaders)
function SIR2(t, u, par)
    # Model parameters.
	N, β, l, β′, β´´, κ, ρ₁, ρ₂,γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ=par
    
    # Current state.
    S, E, I, P, A, H, R, F = u
	
# ODE
    dS = - β * I * S/N - l * β * H * S/N - β′* P * S/N - β´´* A * S/N # susceptible individuals
    dE = β * I * S/N + l * β * H * S/N + β′ *P* S/N + β´´* A * S/N - κ * E # exposed individuals
    dI = κ * ρ₁ * E - (γₐ + γᵢ )*I - δᵢ * I #symptomatic and infectious individuals
    dP = 0
    dA = κ *(1 - ρ₁ - ρ₂ )* E - δₐ*A# infectious but asymptomatic individuals
	dH = γₐ *(I + P ) - γᵣ *H - δₕ *H # hospitalized individuals
	dR = γᵢ * (I + P ) + γᵣ* H # recovery individuals
	dF = δᵢ * I + δₚ* P + δₐ*A + δₕ *H # dead individuals
    return [dS, dE, dI, dP, dA, dH, dR, dF]
end


## optimazation of parameters

function loss_2f8(b)# loss function
	p=copy(par)
	p[5]=b[1]
	p[15]=b[2]
	p[2]=b[3]
	p[7]=b[4]
	
	#initial conditions
	_, x = FDEsolver(SIR2, tspan, X0, ones(8), p, h = .02)
    IPH=vec(sum(x[1:50:end,[3,4,6]], dims=2))
	F=x[1:50:end,8]
    rmsd([C  TrueF], [IPH  F]) # root-mean-square error
end

p_lo=vcat(0,0,.1,0)
p_up=vcat(5,.04,3.5,.6)
pvec=vcat( β´´, δₐ, β, ρ₁)

display("Results for M1:")

Res2F8=optimize(loss_2f8,p_lo,p_up,pvec,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
			Optim.Options(outer_iterations = 3,
						  iterations=20,
						  show_trace=true,
						  show_every=1))

						  
 p2f8=vcat(Optim.minimizer(Res2F8))
par2f8=copy(par); par2f8[5]=p2f8[1]; par2f8[15]=p2f8[2]; 
par2f8[2]=p2f8[3]; par2f8[7]=p2f8[4]

## results

t, x2f8 = FDEsolver(SIR2, tspan, X0, ones(8), par2f8, h = .02) # solve incommensurate fode model

IPH2f8=sum(x2f8[1:50:end,[3,4,6]], dims=2);F2f8=x2f8[1:50:end,8]

Err2f8=rmsd([C  TrueF], [IPH2f8  F2f8]) # RMSE for incommensurate fode model

display(["Err2f8=",Err2f8,"par2f8=",par2f8])
# display([IPH2f8, F2f8])

function myshowall(io, x, limit = false)
  println(io, summary(x), ":")
  Base.print_matrix(IOContext(io, :limit => limit), x)
end

myshowall(stdout, [IPH2f8 F2f8], false)
