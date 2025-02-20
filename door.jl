### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 90391707-4759-4428-8304-78b44db6eee4
using Pkg

# ╔═╡ d93abcf0-199f-4ce4-9616-b7f9fc69b949
Pkg.activate(@__DIR__)

# ╔═╡ 248bf010-7d15-4886-82e5-7fbbd7e0f7ec
using LinearAlgebra, RxInfer, Plots, PGFPlotsX, LogExpFunctions, PlutoUI

# ╔═╡ 0473fbf8-cc8d-486a-8795-8899bd7c68f0
using JLD2

# ╔═╡ 6e6347cf-f7c4-406f-95c8-715c1af62d88
# Multi-Agent Trajectory Planningのdoor.jlの例

# ╔═╡ 8850001f-1a34-42b0-8632-e4a01a6c2ab9
# JLD2に結果を保存

# ╔═╡ a84aa7c1-5b88-40d7-beb9-9a3491eb072d
Pkg.instantiate()

# ╔═╡ 90ef24e0-1fd9-11ee-2ed4-e947901d2a4c
md"""
# Door
"""

# ╔═╡ 5123688d-a83f-4e09-a0f1-e03a1f583264
begin

	ReactiveMP.constrain_form(pmconstraint::RxInfer.PointMassFormConstraint, distribution::MultivariateNormalDistributionsFamily) = mean(distribution) 

	ReactiveMP.prod(::ProdAnalytical, left::Distribution, right::PointMass{Float64}) = right

	softmin(x; l=10) = -logsumexp(-l .* x)/l	

	struct Rectangle
		center
		height
		width
	end

	function distance(r::Rectangle, x) 
		if abs(x[1] - r.center[1]) > r.width/2 || abs(x[2] - r.center[2]) > r.height/2
			# outside of rectangle
			dx = max(abs(x[1] - r.center[1]) - r.width / 2, 0);
			dy = max(abs(x[2] - r.center[2]) - r.height / 2, 0)
			return sqrt(dx^2 + dy^2)
		else
			# inside rectangle
			return max(abs(x[1] - r.center[1]) - r.width/2, abs(x[2] - r.center[2]) - r.height/2)
		end
	end

	distance(env::Tuple, x) = softmin([distance(item, x) for item in env])
	
	function draw_circle!(center, radius; kwargs...)
		θ = range(0, 2π, 100)
		plot!(center[1] .+ radius .* cos.(θ), center[2] .+ radius .* sin.(θ); kwargs...)
	end

	function draw_rectangle!(r::Rectangle; kwargs...)
		plot!(Shape(r.center[1] .+ r.width/2*[-1,1,1,-1], r.center[2] .+ r.height/2 * [-1,-1, 1, 1]); kwargs...)
	end

	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usetikzlibrary{patterns}");
end;

# ╔═╡ bacf9b04-84e8-44cc-bdaa-b8dc1628ff62
md"Save figures $(@bind save_figures CheckBox(default=false))"

# ╔═╡ 8aafd494-661b-4368-993f-29c343c35d0a
if save_figures
	mkpath("exports")
	nothing
end

# ╔═╡ f3b412ba-f0e6-4423-a44d-dcf04ad2b475
md"""
## Problem sketch
"""

# ╔═╡ 156d54ec-126b-4de1-a483-990b503c4a94
begin
	goals = hcat([
		# agent 1: start at (5,-15) with 0 velocity, end at (-5,15) with 0 velocity
		[
			[5, 0, -15, 0],
			[-5, 0, 15, 0]
		], 
		# agent 2: start at (-5,-15) with 0 velocity, end at (5,15) with 0 velocity
		[
			[-5, 0, -15, 0],
			[5, 0, 15, 0]
		],
		# agent 3: start at (-5,15) with 0 velocity, end at (5,-15) with 0 velocity
		[
			[-5, 0, 15, 0],
			[5, 0, -15, 0]
		],
		# agent 4: start at (5,15) with 0 velocity, end at (-5,-15) with 0 velocity
		[
			[5, 0, 15, 0],
			[-5, 0, -15, 0]
		]
	]...)
	radius = 2.5
	environment = (
		Rectangle([-40, 0], 5, 70),
		Rectangle([40, 0], 5, 70)
	)
end;

# ╔═╡ ef868b3a-0616-495e-9bd0-de2edf1089db
begin
	plot(size = (600, 600))

	scatter!([goals[2,1][1]], [goals[2,1][3]], color="red", label="", marker=:star5, markersize=10)
	scatter!([goals[2,2][1]], [goals[2,2][3]], color="blue", label="", marker=:star5, markersize=10)
	scatter!([goals[2,3][1]], [goals[2,3][3]], color="orange", label="", marker=:star5, markersize=10)
	scatter!([goals[2,4][1]], [goals[2,4][3]], color="green", label="", marker=:star5, markersize=10)
	
	draw_circle!([goals[1,1][1], goals[1,1][3]], radius; color="red", label="")
	draw_circle!([goals[1,2][1], goals[1,2][3]], radius; color="blue", label="")
	draw_circle!([goals[1,3][1], goals[1,3][3]], radius; color="orange", label="")
	draw_circle!([goals[1,4][1], goals[1,4][3]], radius; color="green", label="")

	draw_rectangle!(environment[1]; label="", color="black", alpha=0.5)
	draw_rectangle!(environment[2]; label="", color="black", alpha=0.5)

	xlims!(-20, 20)
	ylims!(-20, 20)
end

# ╔═╡ 452ea6ac-17ba-4bd5-8037-ef53b488e420
md"""
## Halfspace prior implementation
"""

# ╔═╡ 60dd5591-cfd1-41c5-a6ed-4c485fa27bb2
begin

	# node specification
	struct Halfspace end
	@node Halfspace Stochastic [out, a, σ2, γ]

	# rule specification
	@rule Halfspace(:out, Marginalisation) (q_a::PointMass, q_σ2::PointMass, q_γ::PointMass) = begin
		return NormalMeanVariance(mean(q_a) + mean(q_γ) * mean(q_σ2), mean(q_σ2))
	end

	@rule Halfspace(:σ2, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, q_a::PointMass, q_γ::PointMass, ) = begin
		return PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_a)) + var(q_out)))
	end
	
end

# ╔═╡ 9638259f-f1bb-4b6e-849d-4455db3a3447
md"""
## Model specification 
"""

# ╔═╡ 2a261390-ad63-47b5-9714-e4a697bc6666
function g(y; r = radius)
	return distance(environment, y) - r
end;

# ╔═╡ b1eec7ac-9ced-4aac-9647-a15322b30fb5
function h(y1, y2, y3, y4; r = radius)
	return softmin([
		norm(y1 - y2) - 2*r,
		norm(y1 - y3) - 2*r,
		norm(y1 - y4) - 2*r,
		norm(y2 - y3) - 2*r,
		norm(y2 - y4) - 2*r,
		norm(y3 - y4) - 2*r,
	])
end;

# ╔═╡ fe9df76f-9df5-46f7-8161-353ddbf94c09
@model function door_model(nr_steps; γ=1, ΔT=1)

	# controls
	u = randomvar(4, nr_steps)

	# hidden state
	x = randomvar(4, nr_steps + 1)

	# observations
	y = randomvar(4, nr_steps)
	goals = datavar(Vector{Int64}, (2, 4))
	
	# distance variable
	d = randomvar(nr_steps)
	dσ2 = randomvar(nr_steps)
	z = randomvar(4, nr_steps)
	zσ2 = randomvar(4, nr_steps)

	# transition model
	A = constvar([1 ΔT 0 0; 0 1 0 0; 0 0 1 ΔT; 0 0 0 1])
	B = constvar([0 0; ΔT 0; 0 0; 0 ΔT])
	C = constvar([1 0 0 0; 0 0 1 0])

	# single agent models
	for k in 1:4

		# prior on state
		x[k,1] ~ MvNormalMeanCovariance(zeros(4), 1e2I)
	
		for t in 1:nr_steps
			
			# prior on controls
			u[k,t] ~ MvNormalMeanCovariance(zeros(2), 1e-1I)
	
			# state transition
			x[k,t+1] ~ A * x[k,t] + B * u[k,t]
	
			# observation model
			y[k,t] ~ C * x[k,t+1]

			# environmental distance
			zσ2[k,t] ~ GammaShapeRate(3/2, γ^2/2)
			z[k,t] ~ g(y[k,t])
			z[k,t] ~ Halfspace(0, zσ2[k,t], γ)

		end

		# goal priors (indexing reverse due to definition)
		goals[1,k] ~ MvNormalMeanCovariance(x[k,1], 1e-5I)
		goals[2,k] ~ MvNormalMeanCovariance(x[k,end], 1e-5I)
		
	end

	# multi-agent models
	for t = 1:nr_steps

		# observation constraint
		dσ2[t] ~ GammaShapeRate(3/2, γ^2/2)
		d[t] ~ h(y[1,t], y[2,t], y[3,t], y[4,t])
		d[t] ~ Halfspace(0, dσ2[t], γ)
		
	end
	
end;

# ╔═╡ 6792be47-148c-414b-9f21-d7ae11807bd9
md"""
## Constraint specification
"""

# ╔═╡ 1e2ba3eb-b073-40cc-8b56-633e18d041d9
@constraints function door_constraints()
	q(d, dσ2) = q(d)q(dσ2)
	q(z, zσ2) = q(z)q(zσ2)
end;

# ╔═╡ e6d347ee-835b-4077-968b-0e7b11540056
md"""
## Probabilistic inference
"""

# ╔═╡ daa54871-05cb-4987-a81c-16f7895da90f
door_meta = @meta begin 
    h() -> Linearization(),
	g() -> Linearization()
end;

# ╔═╡ 12677bf1-5cae-405b-baf0-b3b741ed1770
nr_steps = 50

# ╔═╡ 5dba141f-a4ff-4439-91bc-c7d6d5a3f18e
nr_iterations = 1500

# ╔═╡ ead11ddb-2c4d-47a7-9cf5-483cb40598a0
results = inference(
	model 			= door_model(nr_steps),
	data  			= ( goals = goals, ),
	constraints 	= door_constraints(),
	meta 			= door_meta,
	iterations 		= nr_iterations,
	returnvars 		= KeepLast(), 
	initmarginals 	= ( 
		dσ2 = repeat([PointMass(1)], nr_steps),
		zσ2 = repeat([PointMass(1)], 4, nr_steps),
		u  = repeat([PointMass(0)], nr_steps)
	),
	initmessages = (
		x = MvNormalMeanCovariance(randn(4), 100I),
		y = MvNormalMeanCovariance(randn(2), 100I)
	),
	options = ( limit_stack_depth = 300, )
)

# ╔═╡ 8e994302-4316-4d23-87e2-020883800e65
md"""
## Results
"""

# ╔═╡ 9512a92b-32df-4d9b-8e02-870ade5ef1ba
begin
	if save_figures
		animation = @animate for t in 1:nr_steps
			plot(size = (0.8*600, 600), legend=false)
	
			plot!(
				map(x -> mean(x)[1], results.posteriors[:y][1,1:t]),
				map(x -> mean(x)[2], results.posteriors[:y][1,1:t]);
				color="red", linestyle=:dash
			)
			plot!(
				map(x -> mean(x)[1], results.posteriors[:y][2,1:t]),
				map(x -> mean(x)[2], results.posteriors[:y][2,1:t]);
				color="blue", linestyle=:dash
			)
			plot!(
				map(x -> mean(x)[1], results.posteriors[:y][3,1:t]),
				map(x -> mean(x)[2], results.posteriors[:y][3,1:t]);
				color="orange", linestyle=:dash
			)
			plot!(
				map(x -> mean(x)[1], results.posteriors[:y][4,1:t]),
				map(x -> mean(x)[2], results.posteriors[:y][4,1:t]);
				color="green", linestyle=:dash
			)
		
			draw_circle!(mean(results.posteriors[:y][1,t]), radius; color="red", label="")
			draw_circle!(mean(results.posteriors[:y][2,t]), radius; color="blue", label="")
			draw_circle!(mean(results.posteriors[:y][3,t]), radius; color="orange", label="")
			draw_circle!(mean(results.posteriors[:y][4,t]), radius; color="green", label="")
	
			draw_rectangle!(environment[1]; label="", color="black", alpha=0.5)
			draw_rectangle!(environment[2]; label="", color="black", alpha=0.5)
		
			scatter!([goals[2,1][1]], [goals[2,1][3]], color="red", label="", marker=:star5, markersize=10)
			scatter!([goals[2,2][1]], [goals[2,2][3]], color="blue", label="", marker=:star5, markersize=10)
			scatter!([goals[2,3][1]], [goals[2,3][3]], color="orange", label="", marker=:star5, markersize=10)
			scatter!([goals[2,4][1]], [goals[2,4][3]], color="green", label="", marker=:star5, markersize=10)
			
			xlims!(-20, 20)
			ylims!(-25, 25)
			
		end
		gif(animation, "exports/door.gif", fps = 15)

	else

		@gif for t in 1:nr_steps
			plot(size = (0.8*600, 600), legend=false)
	
			plot!(
				map(x -> mean(x)[1], results.posteriors[:y][1,1:t]),
				map(x -> mean(x)[2], results.posteriors[:y][1,1:t]);
				color="red", linestyle=:dash
			)
			plot!(
				map(x -> mean(x)[1], results.posteriors[:y][2,1:t]),
				map(x -> mean(x)[2], results.posteriors[:y][2,1:t]);
				color="blue", linestyle=:dash
			)
			plot!(
				map(x -> mean(x)[1], results.posteriors[:y][3,1:t]),
				map(x -> mean(x)[2], results.posteriors[:y][3,1:t]);
				color="orange", linestyle=:dash
			)
			plot!(
				map(x -> mean(x)[1], results.posteriors[:y][4,1:t]),
				map(x -> mean(x)[2], results.posteriors[:y][4,1:t]);
				color="green", linestyle=:dash
			)
		
			draw_circle!(mean(results.posteriors[:y][1,t]), radius; color="red", label="")
			draw_circle!(mean(results.posteriors[:y][2,t]), radius; color="blue", label="")
			draw_circle!(mean(results.posteriors[:y][3,t]), radius; color="orange", label="")
			draw_circle!(mean(results.posteriors[:y][4,t]), radius; color="green", label="")
	
			draw_rectangle!(environment[1]; label="", color="black", alpha=0.5)
			draw_rectangle!(environment[2]; label="", color="black", alpha=0.5)
		
			scatter!([goals[2,1][1]], [goals[2,1][3]], color="red", label="", marker=:star5, markersize=10)
			scatter!([goals[2,2][1]], [goals[2,2][3]], color="blue", label="", marker=:star5, markersize=10)
			scatter!([goals[2,3][1]], [goals[2,3][3]], color="orange", label="", marker=:star5, markersize=10)
			scatter!([goals[2,4][1]], [goals[2,4][3]], color="green", label="", marker=:star5, markersize=10)
			
			xlims!(-20, 20)
			ylims!(-25, 25)
			
		end
		
	end
	
end

# ╔═╡ ace972e8-e897-48f4-9b08-e782d20b4b91
# Extract trajectories from results
function extract_trajectories(results, nr_steps)
    trajectories = []
    
    for drone in 1:4
        # Extract positions from results
        positions = [mean(results.posteriors[:y][drone,t]) for t in 1:nr_steps]
        
        # Convert to array format with constant height (z=1.0)
        trajectory = Array{Float64}(undef, nr_steps, 3)
        for (t, pos) in enumerate(positions)
            trajectory[t,:] = [pos[1], pos[2], 1.0]  # Using constant height
        end
        
        push!(trajectories, trajectory)
    end
    
    return trajectories
end

# ╔═╡ 1fc3314e-0b32-4da7-995b-038e6b4181fd
# Save trajectories in JLD2 format
function save_trajectories(trajectories, filename)
    jldopen(filename, "w") do file
        file["trajectories"] = trajectories
        file["num_drones"] = length(trajectories)
        file["timesteps"] = size(trajectories[1], 1)
    end
end

# ╔═╡ 2fa66910-e2dd-475c-9757-c06cce8a4b39
trajectories = extract_trajectories(results, nr_steps)

# ╔═╡ 186c3436-ec6e-4240-8a08-4dbb9b1fd14a
save_trajectories(trajectories, "drone_trajectory.jld2")

# ╔═╡ Cell order:
# ╠═6e6347cf-f7c4-406f-95c8-715c1af62d88
# ╠═8850001f-1a34-42b0-8632-e4a01a6c2ab9
# ╠═90391707-4759-4428-8304-78b44db6eee4
# ╠═d93abcf0-199f-4ce4-9616-b7f9fc69b949
# ╠═a84aa7c1-5b88-40d7-beb9-9a3491eb072d
# ╟─90ef24e0-1fd9-11ee-2ed4-e947901d2a4c
# ╠═248bf010-7d15-4886-82e5-7fbbd7e0f7ec
# ╠═5123688d-a83f-4e09-a0f1-e03a1f583264
# ╠═bacf9b04-84e8-44cc-bdaa-b8dc1628ff62
# ╠═8aafd494-661b-4368-993f-29c343c35d0a
# ╟─f3b412ba-f0e6-4423-a44d-dcf04ad2b475
# ╠═156d54ec-126b-4de1-a483-990b503c4a94
# ╟─ef868b3a-0616-495e-9bd0-de2edf1089db
# ╟─452ea6ac-17ba-4bd5-8037-ef53b488e420
# ╠═60dd5591-cfd1-41c5-a6ed-4c485fa27bb2
# ╟─9638259f-f1bb-4b6e-849d-4455db3a3447
# ╠═2a261390-ad63-47b5-9714-e4a697bc6666
# ╠═b1eec7ac-9ced-4aac-9647-a15322b30fb5
# ╠═fe9df76f-9df5-46f7-8161-353ddbf94c09
# ╟─6792be47-148c-414b-9f21-d7ae11807bd9
# ╠═1e2ba3eb-b073-40cc-8b56-633e18d041d9
# ╟─e6d347ee-835b-4077-968b-0e7b11540056
# ╠═daa54871-05cb-4987-a81c-16f7895da90f
# ╠═12677bf1-5cae-405b-baf0-b3b741ed1770
# ╠═5dba141f-a4ff-4439-91bc-c7d6d5a3f18e
# ╠═ead11ddb-2c4d-47a7-9cf5-483cb40598a0
# ╟─8e994302-4316-4d23-87e2-020883800e65
# ╟─9512a92b-32df-4d9b-8e02-870ade5ef1ba
# ╠═0473fbf8-cc8d-486a-8795-8899bd7c68f0
# ╠═ace972e8-e897-48f4-9b08-e782d20b4b91
# ╠═1fc3314e-0b32-4da7-995b-038e6b4181fd
# ╠═2fa66910-e2dd-475c-9757-c06cce8a4b39
# ╠═186c3436-ec6e-4240-8a08-4dbb9b1fd14a
