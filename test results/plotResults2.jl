
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
dataset_R = CSV.read("time_series_covid19_deaths_global.csv", DataFrame) # all data of Death
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
	N, β, l, β′, β´´, κ, ρ₁, ρ₂,γₐ,	γᵢ,	γᵣ,	δᵢ,	δₚ, δₕ, δₐ=par

    # Current state.
    S, E, I, P, A, H, R, F = u

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

## fitted parameters from Super Computer: CSC_PUHTI

par1= [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 6.767816564146801, 0.11784770556372894, 0.58, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 0.5765375409066592, 1.0000000000000006e-5, 0.9999999999999999, 0.0025773545913321608]

# par1f=    [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 6.767816564146801, 0.11784770556372894, 0.7269816927243573, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 0.01290600743616418, 0.9999999999999999, 0.08758464761573745, 0.0025773545913321608]
# μ1=   0.8927704810509547

par1f8= [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 6.767816564146801, 0.11784770556372894, 0.58, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 0.06693778404993166, 1.0000000000075787e-5, 0.19212213014989807, 0.0025773545913321608]
μ1f8= [0.6822526653627409, 0.9207683038001199, 0.4000000000000001, 0.4000000000000001, 0.9999999999999997, 0.9999999999999999, 0.42946439810615805, 0.7856328235866448]

par2=  [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 1.0000000025526831e-5, 0.11784770556372894, 0.58, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 0.5765782824884094, 1.0000000000065304e-5, 0.9999999999999999, 1.0000000000000296e-5]

# par2f=    [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 19.999999999992536, 0.11784770556372894, 0.5855957331767807, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 1.0000000000000004e-5, 0.9999999999999999, 0.03668149849033601, 0.0024881966891788826]
# μ2=   0.9998999999999999

par2f8= [9491.698589102096, 3.7867885928014893, 2.282972432258884, 0.9407139791767547, 1.000000012966062e-5, 0.11784770556372894, 0.58, 0.001, 0.15459463331892578, 0.6976487340139972, 0.001310080700085496, 1.0000000000000817e-5, 1.0000000000064726e-5, 0.18752664737438562, 0.03850494063794006]
μ2f8=   [0.7090074851163257, 0.8628859483917304, 0.4000000000000001, 0.40000000000000036, 0.9999999999999999, 0.9999999999999999, 0.4313769963746942, 0.47242210916287575]
## Errors

DateTick=Date(2020,3,3):Day(1):Date(2020,5,17)
DateTick2= Dates.format.(DateTick, "d u")

t, x1 = FDEsolver(SIR1, tspan, X0, ones(8), par1, h = .1,nc=4) # solve ode model
# _, x1f = FDEsolver(SIR1, tspan, X0, μ1*ones(8), par1f, h = .1) # solve commensurate fode model
_, x1f8 = FDEsolver(SIR1, tspan, X0, μ1f8, par1f8, h = .1,nc=4) # solve incommensurate fode model
_, x2 = FDEsolver(SIR2, tspan, X0, ones(8), par2, h = .1,nc=4)
# _, x2f = FDEsolver(SIR2, tspan, X0, μ2*ones(8), par2f, h = .1)
_, x2f8 = FDEsolver(SIR2, tspan, X0, μ2f8, par2f8, h = .1,nc=4) # solve incommensurate fode model


IPH1=sum(x1[1:10:end,[3,4,6]], dims=2);F1=x1[1:10:end,8];R1=x1[1:10:end,7]
# IPH1f=sum(x1f[1:10:end,[3,4,6]], dims=2);F1f=x1f[1:10:end,8]
IPH1f8=sum(x1f8[1:10:end,[3,4,6]], dims=2);F1f8=x1f8[1:10:end,8];R1f8=x1f8[1:10:end,7]
IPH2=sum(x2[1:10:end,[3,4,6]], dims=2);F2=x2[1:10:end,8];R2=x2[1:10:end,7]
# IPH2f=sum(x2f[1:10:end,[3,4,6]], dims=2);F2f=x2f[1:10:end,8]
IPH2f8=sum(x2f8[1:10:end,[3,4,6]], dims=2);F2f8=x2f8[1:10:end,8];R2f8=x2f8[1:10:end,7]

#errors from super computer CSC_PUHTI

Err1=117.12639149333518
# Err1f=107.60663030231854
Err1f8= 96.73463457046459
Err2= 95.21981008544995
# Err2f=95.19168968943418
Err2f8=95.07349016009795

#results from super computer CSC_PUHTI ID_Job=16016915 for fractional model 2
# RESULTS=[   5.0                    0.0
#    3.0717655196872515     0.01360324270157735
#    2.999922355993544      0.044853941244534005
#    3.657218423322628      0.09064541020395397
#    4.824490076809029      0.15366247001786962
#    6.514483678468931      0.23970888206968427
#    8.85353874129187       0.35734337751071643
#   12.05030878912861       0.5183806033572562
#   16.39737011448664       0.7388845667890709
#   22.28705247299929       1.0405901692395465
#   30.234130619497236      1.452796385098194
#   40.900619519032375      2.0148099010658203
#   55.11640062438673       2.779011079630416
#   73.88505628492199       3.8145592131834865
#   98.35768336440424       5.211626591774472
#  129.7502161550879        7.085816717136741
#  169.17649035018547       9.582054158974561
#  217.3790037717547       12.876749150996549
#  274.37395556070294      17.176560148774406
#  339.0914714985446       22.711884004355813
#  409.1647346868144       29.723709967470317
#  481.04429355927897      38.44400083921106
#  550.5205045220567       49.0721265566747
#  613.5351234906169       61.752104846151674
#  666.9793713963002       76.55611930958655
#  709.1661836004802       93.47822281961724
#  739.847852775596       112.43894140201945
#  759.8794434733802      133.2983455964733
#  770.7443583792992      155.8734865098789
#  774.1297771887153      179.95620944132676
#  771.6414628106834      205.32860301419444
#  764.6609410959122      231.77482798498238
#  754.3056575985765      259.089218681653
#  741.4467888643109      287.0811936590624
#  726.7506238757288      315.57773870448176
#  710.7235604460164      344.42420016124703
#  693.7515530839003      373.4839908307162
#  676.1312752524982      402.6376521915292
#  658.0933613687139      431.7815772139459
#  639.8192131793055      460.8265908469372
#  621.4529948153034      489.6965093292769
#  603.1101952739233      518.3267486338872
#  584.8838026699126      546.6630197170591
#  566.848833393417       574.6601280186923
#  549.065726316921       602.2808825447399
#  531.5829455964982      629.4951128865364
#  514.4390217914505      656.2787887331208
#  497.66418530115527     682.6132345588605
#  481.28169643218286     708.4844314360818
#  465.30894395072886     733.8823978490302
#  449.7583627002919      758.8006416784339
#  434.6382067817528      783.2356760093096
#  419.9532053255315      807.1865919839978
#  405.7051213937299      830.6546825167517
#  391.89322999366095     853.6431112708075
#  378.5147279052804      876.1566218547857
#  365.5650856038996      898.2012827130956
#  353.03834972723087     919.7842636608502
#  340.92740311649874     940.9136404471288
#  329.2241883393435      961.5982241225217
#  317.91989969907587     981.8474123401342
#  307.0051479969424     1001.671060036329
#  296.470101703341      1021.0793672215511
#  286.3046076832986     1040.0827818654755
#  276.49829419095346    1058.6919160873695
#  267.0406584820311     1076.9174740646617
#  257.92114108111093    1094.7701902526994
#  249.12918847244302    1112.2607766689255
#  240.6543057524657     1129.3998781372109
#  232.4861005826894     1146.1980345147942
#  224.614319609186      1162.66564903686
#  217.02887836488804    1178.8129620137909
#  209.7198855408331     1194.6500292049136
#  202.67766239898367    1210.1867042713495
#  195.89275800050677    1225.4326247804936
#  189.35596083701543    1240.3972012966487]
#
# IPH2f8=RESULTS[:,1]
# F2f8=RESULTS[:,2]
# rmsd([C  TrueF], [IPH2f8  F2f8])

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
 labels=["M1, RMSD=117.13" "FM1, RMSD=96.73" "M2, RMSD=95.22" "FM2, RMSD=95.07"],xrotation=rad2deg(pi/3), linewidth=3)


scatter(DateTick2,TrueR, label= "Real data",legendposition=(.16,.9),
 	title = "(b)" , titleloc = :left, titlefont = font(9), color=:gray22,markerstrokewidth=0)
 	plF=plot!([R1 R1f8 R2 R2f8], ylabel="Cumulative death cases",linestyle=[:solid :dash :dot :dashdot],
 	color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],
  labels=["M1, RMSD=117.13" "FM1, RMSD=96.73" "M2, RMSD=95.22" "FM2, RMSD=95.07"],xrotation=rad2deg(pi/3), linewidth=3)		#  plPortugal=bar!(["M1" "FM1" "M2" "FM2"], [Err1 Err1f8 Err2 Err2f8], ylabel="RMSD",
		# 		legend=:false, bar_width=2,yguidefontsize=8,xtickfontsize=7,
	    # inset = (bbox(.3, .1, 150px, 60px, :left)),
	    # subplot = 2,
	    # bg_inside = nothing)

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

savefig(PlotSEIPAHRF,"PlotSEIPAHRF.svg")


N1=sum(x1,dims=2)
plot(N1)
N2=sum(x2,dims=2)
plot(N2)
N1f8=sum(x1f8,dims=2)
plot(N1f8)
N2f8=sum(x2f8,dims=2)
plot!(N2f8)

pl7=plot(t,[x1[:,7] x1f8[:,7] x2[:,7] x2f8[:,7]] , legend=:false,color=[:slateblue2 :dodgerblue1 :darkorange1 :palegreen3],ylabel="R",xlabel="time (day)")

scatter!(TrueR)
