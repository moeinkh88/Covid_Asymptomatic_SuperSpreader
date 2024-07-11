
# reference for model
# https://doi.org/10.1016/j.chaos.2020.109846
# https://doi.org/10.1016/j.chaos.2021.110652
# https://doi.org/10.3390/axioms10030135

using Optim, StatsBase
using FdeSolver
using DifferentialEquations
using Plots,StatsPlots, StatsPlots.PlotMeasures
using SpecialFunctions
using CSV, HTTP, DataFrames, Dates

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


# Initial conditions
tspan=[1,length(C)] # time span [initial time, final time]

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

S0=10280000/NN-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0


X0=[S0, E0, I0, P0, A0, H0, R0, F0] # initial values
prob = ODEProblem(SIR, X0, tspan, par)

x = solve(prob, alg_hints=[:stiff]; saveat=1)
F=x[8,:]
IPH=x[3,:] .+ x[4,:] .+ x[6,:]
pred=[IPH F]
rmsd([C TrueF], pred)

## fitted parameters from Super Computer: CSC_PUHTI

DateTick=Date(2020,3,3):Day(1):Date(2020,5,17)
DateTick2= Dates.format.(DateTick, "d u")


## Plot
gr()
plot(DateTick2,C,color=:gray22,markerstrokewidth=0,marker=(:circle),linestyle=:dot,
		title = "(a)" , titleloc = :left, titlefont = font(9))
		plC=plot!(IPH, ylabel="Daily new confirmed cases",
		color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],linestyle=[:solid :dash :dot :dashdot],
 legend=:false,xrotation=rad2deg(pi/3), linewidth=3)

scatter(DateTick2,TrueF, label= "Real data",legendposition=(.16,.9),
	title = "(b)" , titleloc = :left, titlefont = font(9), color=:gray22,markerstrokewidth=0)
	plF=plot!(F, ylabel="Cumulative death cases",linestyle=[:solid :dash :dot :dashdot],
	color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],
 labels=["M1, RMSD=117.13" "FM1, RMSD=96.76" "M2, RMSD=95.22" "FM2, RMSD=95.03"],xrotation=rad2deg(pi/3), linewidth=3)

PlPortugal=plot(plC,plF, layout = grid(1,2), size=(700,450))

##S, E, I, P, A, H, R, F
pl1=plot(x.t,x[1,:],ylabel="S")

	pl2=plot(x.t,x[2,:],ylabel="E")

	pl3=plot(x.t,x[3,:],ylabel="I")

	pl4=plot(x.t,x[4,:],ylabel="P")

	pl5=plot(x.t,x[5,:],ylabel="A")

	pl6=plot(x.t,x[6,:],ylabel="H")

	pl7=plot(x.t,x[7,:],
	ylabel="R",xlabel="time (day)")

	pl8=plot(x.t,x[8,:],
		ylabel="F",xlabel="time (day)")



PlotSEIPAHRF=plot(pl1,pl2,pl3,pl4,pl5,pl6,pl7,pl8, linewidth=3, layout = grid(4,2),
	linestyle=[:solid :dash :dot :dashdot],size=(800,570))

# savefig(PlotSEIPAHRF,"PlotSEIPAHRF.svg")

