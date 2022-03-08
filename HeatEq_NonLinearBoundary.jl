

#=
       Heat equation with Nonlinear boundary conditions
Packages LinearAlgebra, SparseArrays, DifferentialEquations, Plots
=#

using DifferentialEquations, Plots 



L = 0.1 			 			# Length of rod
Tf = 60.0 			 			# Final simulation time

λ = 45.0   	 			# Thermal conductivity
ρ = 7800.0 	 			# Mass density
cap = 480.0	 			# Specific heat capacitivity

#=
The diffusivity constant is given by α = λ/(cap * ρ)  
 The heat equation is u`(t,x) = α d^2u(t,x)/dx^2
 u(0,x) = u0(x)
 =#
 
α = λ/(cap * ρ)   			 	# Diffusivity

 
h = 10.0				 	# Heat transfer coefficient
ϵ = 0.6  				 	# Emissivity
sb = 5.67*10^(-8) 		 	# Stefan-Boltzmann constant
k = ϵ * sb 	  			 	# Radiation coefficient

θamb = 298.0 				 	# Ambient temperature in Kelvin
N = 101 						# Number of grid elements
Δx = L/(N-1) 					# Finite discretization   

function diffusion!(dx,x)                 # Diffusion matrix 
	
	for i in 2:length(x)-1
	  dx[i] = x[i-1] - 2x[i] + x[i+1]        #Talyor series
	end
	dx[1] = -2x[1] + 2x[2]                # Boundary conditions
	dx[end] = 2x[end-1] - 2x[end]
  end
  


function heat_eq!(dθ, θ, p, t)   #Heat equation as ODE
	
	N = size(θ)
	Φout = zeros(N)
	
	Φout[1] = -h * (θ[1] - θamb) - k*(θ[1]^4 - θamb^4) 				#The sum of heat radiation and heat transfer

	Φout[end] = -h * (θ[end] - θamb) - k*(θ[end]^4 - θamb^4)

	#the spatial approximation the heat equation

	diffusion!(dθ,θ) 													 
    dθ .= (1/Δx^2) * α * dθ                                              
	dθ[1] = dθ[1] + (2/Δx) * α/λ * Φout[1]
	dθ[end] = dθ[end] + (2/Δx) * α/λ * Φout[end]

end

#Initial Conditions

θ₀ = 10^3 * ones(N) 			# initial data of the heat equation is assumed as Uo(x) = 1000°K
Δt_ul = (0.5*Δx^2)/α  				 #Condtion of stability
Δt = 10^(-2) # Sampling time

#Simulation

tspan = (0.0, Tf)
xspan = 0 : Δx : L 					# 1-dimensional grid
#param = [α]

prob = ODEProblem(heat_eq!, θ₀, tspan) # ODE Problem
sol = solve(prob,Euler(),dt=Δt,progress=true, save_everystep=true, save_start=true) #Solving the ODE


#prob_func = (prob,i,repeat) -> remake(prob,θ₀=rand(101).*θ₀, param=rand(101).*param )
#monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)
#sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=100_000,saveat=1.0f0)


p_Temp= plot(xspan, sol.u[end], 
             xlabel = "Position x",
             ylabel="Temperature", 
             title="Temperature at the final time", 
            legend=false)

savefig(p_Temp, "Temperature_gr.png")

h_Evel = heatmap(sol.t, xspan, 
				 sol[1:end,1:end], 
				 xaxis="Time [s]",
				 yaxis="Position [m]", 
				 title="Evolution of temperature")

savefig(h_Evel, "Evolution_gr.png")
