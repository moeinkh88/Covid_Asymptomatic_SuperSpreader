
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

par=[10280000/NN,  β, l, β′, β´´, κ, ρ₁, ρ₂,	γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ] # parameters


# Initial conditions
N=10280000/NN # Population Size
S0=N-5; E0=0; I0=4; P0=1; A0=0; H0=0; R0=0; F0=0
X0=[N-5, E0, I0, P0, A0, H0, R0, F0] # initial values
tspan=[1,length(C)] # time span [initial time, final time]

function SIR_M1(t, u, par)
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

function SIR_M2(t, u, par)
    # Model parameters.
	N, β, l, β′, β´´, κ, ρ₁, ρ₂,γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ=par

    # Current state.
    S, E, I, P, A, H, R, F = u

# ODE
dS = - β * I * S/N - l * β * H * S/N - β′* P * S/N - β´´* A * S/N # susceptible individuals
dE = β * I * S/N + l * β * H * S/N + β′ *P* S/N + β´´* A * S/N - κ * E # exposed individuals
    dI = κ * ρ₁ * E - (γₐ + γᵢ )*I - δᵢ * I #symptomatic and infectious individuals
    dP = κ* (1-ρ₁) * E - (γₐ + γᵢ)*P - δₚ * P # super-spreaders individuals
    dA = 0
	dH = γₐ *(I + P ) - γᵣ *H - δₕ *H # hospitalized individuals
	dR = γᵢ * (I + P ) + γᵣ* H # recovery individuals
	dF = δᵢ * I + δₚ* P + δₐ*A + δₕ *H # dead individuals
    return [dS, dE, dI, dP, dA, dH, dR, dF]
end
function SIR_M3(t, u, par)
    # Model parameters.
	N, β, l, β′, β´´, κ, ρ₁, ρ₂,γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ=par

    # Current state.
    S, E, I, P, A, H, R, F = u

# ODE
dS = - β * I * S/N - l * β * H * S/N - β′* P * S/N - β´´* A * S/N # susceptible individuals
dE = β * I * S/N + l * β * H * S/N + β′ *P* S/N + β´´* A * S/N - κ * E # exposed individuals
    dI = κ * ρ₁ * E - (γₐ + γᵢ )*I - δᵢ * I #symptomatic and infectious individuals
    dP = κ* ρ₂ * E - (γₐ + γᵢ)*P - δₚ * P # super-spreaders individuals
    dA = κ *(1 - ρ₁ - ρ₂ )* E - δₐ*A# infectious but asymptomatic individuals
	dH = γₐ *(I + P ) - γᵣ *H - δₕ *H # hospitalized individuals
	dR = γᵢ * (I + P ) + γᵣ* H # recovery individuals
	dF = δᵢ * I + δₚ* P + δₐ*A + δₕ *H # dead individuals
    return [dS, dE, dI, dP, dA, dH, dR, dF]
end


# M1
# "Err2f8="
# 123.30006541230019
#    "par2f8="
par_M1=[7542.62462248916, 3.368739014223833, 1.605375275205752, 0.0, 4.9999999999995355, 0.03658959949393572, 0.6999999999999992, 0.0, 0.1375531965076037, 0.0560295261424954, 0.9179720534039433, 0.044242996045119866, 0.0, 0.003473250106648919, 0.005552315473508071];

# M2
# "Err2f8="
# 116.40553143270077
#    "par2f8="
par_M2=[7542.62462248916, 3.12996077117126, 1.605375275205752, 7.999999997998878, 0.0, 0.03658959949393572, 0.8760560896249793, 0.25088774743033504, 0.1375531965076037, 0.0560295261424954, 0.9179720534039433, 0.044242996045119866, 0.019687597179891277, 0.003473250106648919, 0.0];

# M3
# "Err2f8="
#   115.23024462015562
# β′ = 4.9724275315175905
#    "par2f8="
par_M3=par;
# par_M3[4]=7.895865150500946


# FM1
# "Err2f8="
# 107.45339388594506
#    "par2f8="
par_FM1=[7542.62462248916, 3.4999999999999996, 1.605375275205752, 0.0, 4.999999999999999, 0.03658959949393572, 0.5999999999999999, 0.0, 0.1375531965076037, 0.0560295261424954, 0.9179720534039433, 0.044242996045119866, 0.0, 0.003473250106648919, 0.00612053486309477]
#    "order2f8="
Order_FM1=[0.903490717147999, 0.9999999999999999, 0.9999999999999999, 0.8500027425706432, 0.9999999999999999, 0.9999999999999989, 0.8500027425706432, 0.8998009940318135]

# FM2
# "Err2f8="
# 109.81476181208696
#    "par2f8="
par_FM2=[7542.62462248916, 3.4999999999999143, 1.605375275205752, 7.740335211476434, 0.0, 0.03658959949393572, 0.8310642637854406, 0.25088774743033504, 0.1375531965076037, 0.0560295261424954, 0.9179720534039433, 0.044242996045119866, 0.039999999999999966, 0.003473250106648919, 0.0]
#    "order2f8="
Order_FM2=[0.9999999999999877, 0.9775812315315028, 0.8931700288111524, 0.9999999999999949, 0.8499099633913844, 0.9999999999999954, 0.8499099633913844, 0.9999999999999999]

# FM3
# "Err2f8="
# 99.39828456428222
# β′ = 7.895865150500946
#   "par2f8="
par_FM3=[7542.62462248916, 2.9763470555591796, 1.605375275205752, 7.895865150500946, 2.77222207964141, 0.03658959949393572, 0.5272699408450209, 0.25088774743033504, 0.1375531965076037, 0.0560295261424954, 0.9179720534039433, 0.044242996045119866, 0.001724702577525102, 0.003473250106648919, 0.02042169878603325]
#   "order2f8="
Order_FM3=[0.9785857005003694, 0.999999999531513, 0.9999999869472375, 0.9999999996031644, 0.999999999151962, 0.9999999974787451, 0.9251790257359261, 0.9398986700128771]

## Errors
t, xM1 = FDEsolver(SIR_M1, tspan, X0, ones(8), par_M1, h = .1,nc=4) # solve ode model M1
t, xM2 = FDEsolver(SIR_M2, tspan, X0, ones(8), par_M2, h = .1,nc=4) # solve ode model M2
t, xM3 = FDEsolver(SIR_M3, tspan, X0, ones(8), par_M3, h = .1,nc=4) # solve ode model M3

_, xFM1= FDEsolver(SIR_M1, tspan, X0, Order_FM1, par_FM1, h = .1,nc=4) # solve incommensurate fode model
_, xFM2= FDEsolver(SIR_M2, tspan, X0, Order_FM2, par_FM2, h = .1,nc=4) # solve incommensurate fode model
_, xFM3= FDEsolver(SIR_M3, tspan, X0, Order_FM3, par_FM3, h = .1,nc=4) # solve incommensurate fode model

IPH_M1=sum(xM1[1:10:end,[3,4,6]], dims=2);DM1=xM1[1:10:end,8];
IPH_M2=sum(xM2[1:10:end,[3,4,6]], dims=2);DM2=xM2[1:10:end,8];
IPH_M3=sum(xM3[1:10:end,[3,4,6]], dims=2);DM3=xM3[1:10:end,8];
IPH_FM1=sum(xFM1[1:10:end,[3,4,6]], dims=2);DFM1=xFM1[1:10:end,8];
IPH_FM2=sum(xFM2[1:10:end,[3,4,6]], dims=2);DFM2=xFM2[1:10:end,8];
IPH_FM3=sum(xFM3[1:10:end,[3,4,6]], dims=2);DFM3=xFM3[1:10:end,8];


# errors from my computer

ErrM1= rmsd([C  TrueF], [IPH_M1  DM1])
ErrM2= rmsd([C  TrueF], [IPH_M2  DM2])
ErrM3= rmsd([C  TrueF], [IPH_M3  DM3])
ErrFM1= rmsd([C  TrueF], [IPH_FM1  DFM1])
ErrFM2= rmsd([C  TrueF], [IPH_FM2  DFM2])
ErrFM3= rmsd([C  TrueF], [IPH_FM3  DFM3])



## Plot

DateTick=Date(2020,3,3):Day(1):Date(2020,5,17)
DateTick2= Dates.format.(DateTick, "d u")

gr()
plot(DateTick2,C,color=:gray22,markerstrokewidth=0,marker=(:circle),linestyle=:dot,
		title = "(a)" , titleloc = :left, titlefont = font(9))
		plC=plot!([IPH_M1 IPH_M2 IPH_M3 IPH_FM1 IPH_FM2 IPH_FM3], ylabel="Daily new confirmed cases",
		color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],linestyle=[:solid :dash :dot :dashdot],
 legend=:false,xrotation=rad2deg(pi/3), linewidth=3)

scatter(DateTick2,TrueF, label= "Real data",legendposition=(.16,.9),
	title = "(b)" , titleloc = :left, titlefont = font(9), color=:gray22,markerstrokewidth=0)
	plF=plot!([F1 F1f8 F2 F2f8], ylabel="Cumulative death cases",linestyle=[:solid :dash :dot :dashdot],
	color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],
 labels=["M1, RMSD=112.85" "FM1, RMSD=96.25" "M2, RMSD=95.19" "FM2, RMSD=95.02"],xrotation=rad2deg(pi/3), linewidth=3)

PlPortugal=plot(plC,plF, layout = grid(1,2), size=(700,450))

savefig(PlPortugal,"PlPortugal.svg")
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

# savefig(PlotSEIPAHRF,"PlotSEIPAHRF.svg")

#plot population
N1=sum(x1[1:10:end,:],dims=2)
N2=sum(x2[1:10:end,:],dims=2)
N1f8=sum(x1f8[1:10:end,:],dims=2)
N2f8=sum(x2f8[1:10:end,:],dims=2)

PlPop=plot(DateTick2,[N1 N1f8 N2 N2f8], color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],linewidth=[6 3 3 3],
		linestyle=[:solid :dash :dot :dashdot], legendposition=(.2,.3),
		legendfont=font(10), labels=["M1" "FM1" "M2" "FM2"],ylabel="N (involved population)")

# savefig(PlPop,"PlPop.svg")
