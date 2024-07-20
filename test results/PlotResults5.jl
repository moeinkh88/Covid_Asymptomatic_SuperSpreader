
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
X01=[N-4, E0, I0, P0-1, A0, H0, R0, F0] # initial values
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
#  108.20699296061734
#    "par2f8="
par_M3=  [7542.62462248916, 3.1220651706673066, 1.605375275205752, 5.079230988158643, 4.6830693978875235, 0.03658959949393572, 0.5999999999989443, 0.14999999999962524, 0.1375531965076037, 0.0560295261424954, 0.9179720534039433, 0.044242996045119866, 0.0010000001259533725, 0.003473250106648919, 0.009696038544937498];


# FM1
# "Err2f8="
# 107.45339388594506
#    "par2f8="
par_FM1=[7542.62462248916, 3.4999999999999996, 1.605375275205752, 0.0, 4.999999999999999, 0.03658959949393572, 0.5999999999999999, 0.0, 0.1375531965076037, 0.0560295261424954, 0.9179720534039433, 0.044242996045119866, 0.0, 0.003473250106648919, 0.00612053486309477]
#    "order2f8="
Order_FM1=[0.903490717147999, 0.9999999999999999, 0.9999999999999999, 0.8500027425706432, 0.9999999999999999, 0.9999999999999989, 1, 0.8998009940318135]

# FM2
# "Err2f8="
# 109.81476181208696
#    "par2f8="
par_FM2=[7542.62462248916, 3.4999999999999143, 1.605375275205752, 7.740335211476434, 0.0, 0.03658959949393572, 0.8310642637854406, 0.25088774743033504, 0.1375531965076037, 0.0560295261424954, 0.9179720534039433, 0.044242996045119866, 0.039999999999999966, 0.003473250106648919, 0.0]
#    "order2f8="
Order_FM2=[0.9999999999999877, 0.9775812315315028, 0.8931700288111524, 0.9999999999999949, 0.8499099633913844, 0.9999999999999954, 1, 0.9999999999999999]

# FM3
# "Err2f8="
# 99.39828456428222
#   "par2f8="
par_FM3=[7542.62462248916, 2.9763470555591796, 1.605375275205752, 7.895865150500946, 2.77222207964141, 0.03658959949393572, 0.5272699408450209, 0.25088774743033504, 0.1375531965076037, 0.0560295261424954, 0.9179720534039433, 0.044242996045119866, 0.001724702577525102, 0.003473250106648919, 0.02042169878603325]
#   "order2f8="
Order_FM3=[0.9785857005003694, 0.999999999531513, 0.9999999869472375, 0.9999999996031644, 0.999999999151962, 0.9999999974787451, 1, 0.9398986700128771]

## Errors
t, xM1 = FDEsolver(SIR_M1, tspan, X01, ones(8), par_M1, h = .1,nc=4); # solve ode model M1
t, xM2 = FDEsolver(SIR_M2, tspan, X0, ones(8), par_M2, h = .1,nc=4); # solve ode model M2
t, xM3 = FDEsolver(SIR_M3, tspan, X0, ones(8), par_M3, h = .1,nc=4); # solve ode model M3

_, xFM1= FDEsolver(SIR_M1, tspan, X01, Order_FM1, par_FM1, h = .1,nc=4); # solve incommensurate fode model
_, xFM2= FDEsolver(SIR_M2, tspan, X0, Order_FM2, par_FM2, h = .1,nc=4); # solve incommensurate fode model
_, xFM3= FDEsolver(SIR_M3, tspan, X0, Order_FM3, par_FM3, h = .1,nc=4); # solve incommensurate fode model

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
ColorPalette=[:magenta2 :darkgoldenrod1 :cyan4 :blueviolet :orangered1 :lime]
plot(DateTick2,C,color=:gray22,markerstrokewidth=0,marker=(:circle),linestyle=:dot,
 title = "(a)" , titleloc = :left, titlefont = font(9),xrotation=rad2deg(pi/3))
 plC=plot!([IPH_M1 IPH_M2 IPH_M3 IPH_FM1 IPH_FM2 IPH_FM3], ylabel="Daily new confirmed cases",
		color=ColorPalette,linestyle=[:dash :dash :dash :solid :solid :solid],
 legend=:false, linewidth=2.6)

scatter(DateTick2,TrueF, label= "Real data",legendposition=(.16,.9),
	title = "(b)" , titleloc = :left, titlefont = font(9), color=:gray22,markerstrokewidth=0)
	plF=plot!([DM1 DM2 DM3 DFM1 DFM2 DFM3],	 ylabel="Cumulative death cases",linestyle=[:dash :dash :dash :solid :solid :solid ],
	color=ColorPalette,
 labels=["M1, RMSD=$(round(ErrM1,digits=3))" "M2, RMSD=$(round(ErrM2,digits=3))" "M3, RMSD=$(round(ErrM3,digits=3))" "FM1, RMSD=$(round(ErrFM1,digits=3))" "FM2, RMSD=$(round(ErrFM2,digits=3))" "FM3, RMSD=$(round(ErrFM3,digits=3))"],xrotation=rad2deg(pi/3), linewidth=2)

PlPortugal=plot(plC,plF, layout = grid(1,2), size=(750,500))

# savefig(PlPortugal,"PlPortugal.svg")
##S, E, I, P, A, H, R, F
pl1=plot(t,[xM1[:,1] xM2[:,1] xM3[:,1]  xFM1[:,1] xFM2[:,1] xFM3[:,1]], color=ColorPalette,  
labels=["M1, RMSD=$(round(ErrM1,digits=3))" "M2, RMSD=$(round(ErrM2,digits=3))" "M3, RMSD=$(round(ErrM3,digits=3))" "FM1, RMSD=$(round(ErrFM1,digits=3))" "FM2, RMSD=$(round(ErrFM2,digits=3))" "FM3, RMSD=$(round(ErrFM3,digits=3))"],ylabel="S")

	pl2=plot(t,[xM1[:,2] xM2[:,2] xM3[:,2]  xFM1[:,2] xFM2[:,2] xFM3[:,2]],color=ColorPalette, legend=:false,ylabel="E")

	pl3=plot(t,[xM1[:,3] xM2[:,3] xM3[:,3]  xFM1[:,3] xFM2[:,3] xFM3[:,3]],color=ColorPalette, legend=:false,ylabel="I")

	pl4=plot(t,[xM1[:,4] xM2[:,4] xM3[:,4]  xFM1[:,4] xFM2[:,4] xFM3[:,4]],color=ColorPalette,   legend=:false,ylabel="P")

	pl5=plot(t,[xM1[:,5] xM2[:,5] xM3[:,5]  xFM1[:,5] xFM2[:,5] xFM3[:,5]],  legend=:false,ylabel="A",color=ColorPalette)

	pl6=plot(t,[xM1[:,6] xM2[:,6] xM3[:,6]  xFM1[:,6] xFM2[:,6] xFM3[:,6]], legend=:false,ylabel="H",color=ColorPalette)

	pl7=plot(t,[xM1[:,7] xM2[:,7] xM3[:,7]  xFM1[:,7] xFM2[:,7] xFM3[:,7]], legend=:false,color=ColorPalette,
	ylabel="R",xlabel="time (day)")

	pl8=plot(t,[xM1[:,8] xM2[:,8] xM3[:,8]  xFM1[:,8] xFM2[:,8] xFM3[:,8]] ,  legend=:false,color=ColorPalette,
		ylabel="F",xlabel="time (day)")



PlotSEIPAHRF=plot(pl1,pl2,pl3,pl4,pl5,pl6,pl7,pl8, linewidth=3, layout = grid(4,2),
	linestyle=[:dash :dash :dash :solid :solid :solid],size=(850,670))

# savefig(PlotSEIPAHRF,"PlotSEIPAHRF.svg")

#plot populationN1=sum(xM1[1:10:end,:],dims=2)
N2=sum(xM2[1:10:end,:],dims=2)
N3=sum(xM3[1:10:end,:],dims=2)
NF1=sum(xFM1[1:10:end,:],dims=2)
NF2=sum(xFM2[1:10:end,:],dims=2)
NF3=sum(xFM3[1:10:end,:],dims=2)

PlPop=plot(DateTick2,[N1 N2 N3 NF1 NF2 NF3], #color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],
 	linewidth=[6 3 3 3],
		linestyle=[:solid :dash :dot :dashdot], legendposition=(.2,.3),
		legendfont=font(10), labels=["M1" "M2" "M3" "FM1" "FM2" "FM3"],ylabel="N (involved population)")

# savefig(PlPop,"PlPop.svg")
