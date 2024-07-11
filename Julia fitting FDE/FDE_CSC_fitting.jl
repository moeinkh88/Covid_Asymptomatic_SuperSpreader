# reference for model
# https://doi.org/10.1016/j.chaos.2020.109846
# https://doi.org/10.1016/j.chaos.2021.110652
# https://doi.org/10.3390/axioms10030135
# we want to fit \beta (3) and \delta_a and orders

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
pp=[2.8063282642844243
6.465045581605194
4.5620373248101345
1130.8386854783253
0.039540872185940025
2.122681091272686
0.9282608821100266
0.13897756053560484
0.23820479628557029
0.053241741371967605
0.01327654134856475
0.010852133061676948
0.01838901519151417
0.2531264602772149
0.554057903606213]
β, β′, β´´, NN, κ, l, γₐ, γᵢ, γᵣ, δᵢ, δₚ, δₕ, δₐ, ρ₂, ρ₁ = pp[1:15]

par=[10280000/NN,  β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ] # parameters


# Initial conditions
N=10280000/NN # Population Size
S0=N-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
X0=[N-5, E0, I0, P0, A0, H0, R0, F0] # initial values

tspan=[1,length(C)] # time span [initial time, final time]

# Define model 1: with super spreaders

# Define model 12: with super spreaders + asymptomatic coefficients
function SIR2(t, u, par)
    # Model parameters.
	N, β, l, β′, β´´, κ, ρ₁, ρ₂,γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ=par

    # Current state.
    S, E, I, P, A, H, R, F = u

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

## optimazation of parameters and order of incommensurate fractional order model2

function loss_2f8(b)# loss function
	p=copy(par)
	p[4:5]=b[1:2]
    p[15]=b[3]
	order = b[4:11] # order of derivatives
	if size(X0,2) != Int64(ceil(maximum(order))) # to prevent any errors regarding orders higher than 1
		indx=findall(x-> x>1, order)
		order[indx]=ones(length(indx))
	end
	# initial conditions
	_, x = FDEsolver(SIR2, tspan, X0, order, p, h = .02)
    # _, x = FDEsolver(SIR2, tspan, X0, ones(8), p, h = .02)
    IPH=vec(sum(x[1:50:end,[3,4,6]], dims=2))
	F=x[1:50:end,8]
    rmsd([C  TrueF], [IPH  F]) # root-mean-square error
end


# Define a penalty function for the constraints
# function penalty_function(p)
#     β, β′, β´´, ρ₂, ρ₁ = p[1:5]
#     penalty = 0.0
    
#     # Add penalties for each constraint violation
#     if β′ < 2 * β 
#         penalty += 300   # Large penalty if β′ is less than 2 * β
#     end
#     # Add other constraints as needed
#     if β´´ >  β || β´´ <  2* β 
#         penalty += 300   
#     end
#     if ρ₁ > 2*ρ₂
#         penalty += 300  
#     end
#     return penalty
# end

# Combine the objective function with the penalty function
# function constrained_loss_2f8(p)
#     return loss_2f8(p) + penalty_function(p)
# end

p_lo=vcat(2*β,β,0,.7*ones(8))
p_up=vcat(10,2*β,.5*δᵢ,ones(8))
pvec=vcat(β′, β´´,.02,.9*ones(8))
# p_lo=vcat(0.1,0,0,0,0,0)
# p_up=vcat(3.3,10,7,.29,.6,.03)
# pvec=vcat(β, β′, β´´, ρ₂, .5,.02)
Res2F8=optimize(loss_2f8,p_lo,p_up,pvec,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
			Optim.Options(outer_iterations = 10,
						 iterations=200,
						  show_trace=true,
						  show_every=1))
p2f8=vcat(Optim.minimizer(Res2F8))
par2f8=copy(par); par2f8[5]=p2f8[1]; par2f8[12:15]=p2f8[2:5]; μ2f8=p2f8[6:13]


## results

_, x2f8 = FDEsolver(SIR2, tspan, X0, μ2f8, par2f8, h = .02) # solve incommensurate fode model

IPH2f8=sum(x2f8[1:50:end,[3,4,6]], dims=2);F2f8=x2f8[1:50:end,8]

Err2f8=rmsd([C  TrueF], [IPH2f8  F2f8]) # RMSE for incommensurate fode model

display(["Err2f8=",Err2f8,"par2f8=",par2f8,"order2f8=",μ2f8])
# display([IPH2f8, F2f8])

function myshowall(io, x, limit = false)
  println(io, summary(x), ":")
  Base.print_matrix(IOContext(io, :limit => limit), x)
end

myshowall(stdout, [IPH2f8 F2f8], false)
