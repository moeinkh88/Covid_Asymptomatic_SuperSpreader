
# reference for model
# https://doi.org/10.1016/j.chaos.2020.109846
# https://doi.org/10.1016/j.chaos.2021.110652
# https://doi.org/10.3390/axioms10030135

using Optim, StatsBase
using FdeSolver
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
#
dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Death
RData=dataset_R[dataset_R[!,2].=="Portugal",45:120]
TrueR=(Float64.(Vector(RData[1,:])))


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
N=10280000/NN # Population Size

par=[N, β, l, β′, β´´, κ, ρ₁,	ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ] # parameters

# Initial conditions

S0=N-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
X0=[S0, E0, I0, P0, A0, H0, R0, F0]
tspan=[1,length(C)] # time span [initial time, final time]

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

## fitted parameters from Super Computer: CSC_PUHTI

par1= [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 6.767816564146801, 0.11784770556372894, 0.6928018382901505, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 0.0010807082772655262, 0.45500168006745334, 0.035917596817983934, 0.0025773545913321608]

# par1f=    [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 6.767816564146801, 0.11784770556372894, 0.7269816927243573, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 0.01290600743616418, 0.9999999999999999, 0.08758464761573745, 0.0025773545913321608]
# μ1=   0.8927704810509547

par1f8=  [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 6.767816564146801, 0.11784770556372894, 0.6850422372504599, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 0.0010807082772655262, 0.45500168006745334, 0.035917596817983934, 0.0025773545913321608]
μ1f8=  [0.9858953150545143, 0.9999999999999992, 0.8512198811452514, 0.4000000000000007, 0.9999999999999868, 0.9999999999999998, 0.9999999999999947, 0.9999999999999999]

par2=   [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 9.999999999369484, 0.11784770556372894, 0.582479682495226, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 0.0010807082772655262, 0.45500168006745334, 0.035917596817983934, 0.0025100631211294384]

# par2f=    [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 19.999999999992536, 0.11784770556372894, 0.5855957331767807, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 1.0000000000000004e-5, 0.9999999999999999, 0.03668149849033601, 0.0024881966891788826]
# μ2=   0.9998999999999999

par2f8= [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 9.99999999999994, 0.11784770556372894, 0.5783344596826828, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 0.0010807082772655262, 0.45500168006745334, 0.035917596817983934, 0.0024803469944495917]
μ2f8= [0.9989254807681097, 0.9999999999999999, 0.9999999999999998, 0.9999999999999996, 0.9999999999999996, 0.9999999999999996, 0.90000000000000036, 0.9999999999999999]
## Errors

DateTick=Date(2020,3,3):Day(1):Date(2020,5,17)
DateTick2= Dates.format.(DateTick, "d u")

t, x1 = FDEsolver(SIR1, tspan, X0, ones(8), par1, h = .1,nc=4) # solve ode model
# _, x1f = FDEsolver(SIR1, tspan, X0, μ1*ones(8), par1f, h = .1) # solve commensurate fode model
_, x1f8 = FDEsolver(SIR1, tspan, X0, μ1f8, par1f8, h = .1,nc=4) # solve incommensurate fode model
_, x2 = FDEsolver(SIR2, tspan, X0, ones(8), par2, h = .1,nc=4)
# _, x2f = FDEsolver(SIR2, tspan, X0, μ2*ones(8), par2f, h = .1)
_, x2f8 = FDEsolver(SIR2, tspan, X0, μ2f8, par2f8, h = .1,nc=4) # solve incommensurate fode model

IPH1=sum(x1[1:10:end,[3,4,6]], dims=2);F1=x1[1:10:end,8]
# IPH1f=sum(x1f[1:10:end,[3,4,6]], dims=2);F1f=x1f[1:10:end,8]
IPH1f8=sum(x1f8[1:10:end,[3,4,6]], dims=2);F1f8=x1f8[1:10:end,8]
IPH2=sum(x2[1:10:end,[3,4,6]], dims=2);F2=x2[1:10:end,8]
# IPH2f=sum(x2f[1:10:end,[3,4,6]], dims=2);F2f=x2f[1:10:end,8]
IPH2f8=sum(x2f8[1:10:end,[3,4,6]], dims=2);F2f8=x2f8[1:10:end,8]

#errors from super computer CSC_PUHTI

Err1= 145.6549785531883
Err1f8=142.55107036468087
Err2= 95.27357699271133
Err2f8=95.19602406416122


## Plot
gr()
plot(DateTick2,C,color=:gray22,markerstrokewidth=0,marker=(:circle),linestyle=:dot,
		title = "(a)" , titleloc = :left, titlefont = font(9))
		plC=plot!([IPH1 IPH1f8 IPH2 IPH2f8], ylabel="Daily new confirmed cases",
		color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],linestyle=[:solid :dash :dot :dashdot],
 legend=:false,xrotation=rad2deg(pi/3), linewidth=3)

scatter(DateTick2,TrueF, label= "Real data",legendposition=(.16,.9),
	title = "(b)" , titleloc = :left, titlefont = font(9), color=:gray22,markerstrokewidth=0)
	plF=plot!([F1 F1f8 F2 F2f8], ylabel="Cumulative death cases",linestyle=[:solid :dash :dot :dashdot],
	color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],
 labels=["M1, RMSD=142.65" "FM1, RMSD=142.55" "M2, RMSD=95.27" "FM2, RMSD=95.20"],xrotation=rad2deg(pi/3), linewidth=3)

PlPortugal=plot(plC,plF, layout = grid(1,2), size=(700,450))

savefig(PlPortugal,"PlPortugal2.svg")
##S, E, I, P, A, H, R, F
pl1=plot(t,[x1[:,1] x1f8[:,1] x2[:,1] x2f8[:,1]], color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],  labels=["M1" "FM1" "M2" "FM2"],ylabel="S")

	pl2=plot(t,[x1[:,2] x1f8[:,2] x2[:,2] x2f8[:,2]],color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3], legend=:false,ylabel="E")

	pl3=plot(t,[x1[:,3] x1f8[:,3] x2[:,3] x2f8[:,3]],color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3], legend=:false,ylabel="I")

	pl4=plot(t,[x1[:,4] x1f8[:,4] x2[:,4] x2f8[:,4]],color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],   legend=:false,ylabel="P")

	pl5=plot(t,[x1[:,5] x1f8[:,5] x2[:,5] x2f8[:,5]],  legend=:false,ylabel="A",color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3])

	pl6=plot(t,[x1[:,6] x1f8[:,6] x2[:,6] x2f8[:,6]], legend=:false,ylabel="H",color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3])

	pl7=plot(t,[x1[:,7] x1f8[:,7] x2[:,7] x2f8[:,7]] , legend=:false,color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],
	ylabel="R",xlabel="time (day)")

	pl8=plot(t,[x1[:,8] x1f8[:,8] x2[:,8] x2f8[:,8]] ,  legend=:false,color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],
		ylabel="F",xlabel="time (day)")



PlotSEIPAHRF=plot(pl1,pl2,pl3,pl4,pl5,pl6,pl7,pl8, linewidth=3, layout = grid(4,2),
	linestyle=[:solid :dash :dot :dashdot],size=(800,570))

savefig(PlotSEIPAHRF,"PlotSEIPAHRF2.svg")

#plot population
N1=sum(x1[1:10:end,:],dims=2)
N2=sum(x2[1:10:end,:],dims=2)
N1f8=sum(x1f8[1:10:end,:],dims=2)
N2f8=sum(x2f8[1:10:end,:],dims=2)

PlPop=plot(DateTick2,[N1 N1f8 N2 N2f8], color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],linewidth=[6 3 3 3],
		linestyle=[:solid :dash :dot :dashdot], legendposition=(.2,.3),
		legendfont=font(10), labels=["M1" "FM1" "M2" "FM2"],ylabel="N (involved population)")

savefig(PlPop,"PlPop2.svg")
