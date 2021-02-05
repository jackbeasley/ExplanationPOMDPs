include("experiment_base.jl")
using ExplanationPOMDPs.Beliefs
using Printf
using QMDP
import DataFrames
import Arrow
import Dates
##

# The goal of this notebook is to compare different selection thresholds at
# different difficulty levels
thresholds = [0.0, 0.3, 0.6, 0.9]

solver = QMDPSolver(max_iterations=20,
                    belres=1e-3,
                    verbose=true
                   ) 

balls_per_observation_range = 5:100

bayes_params = vec([
    (pomdp, 
        BeliefThresholdPolicy(pomdp, threshold, -1), 
        DiscreteUpdater(pomdp),
        Dict{Symbol,Any}(:policy => (@sprintf "Threshold %s" threshold), :rule => "Bayes")) 
    for pomdp in [ 
        standard_reward_pomdp(10, balls_per_observation)
        for balls_per_observation in balls_per_observation_range
    ], threshold in thresholds
])

bayes_optimal_params = vec([
    (pomdp, 
        solve(solver, pomdp),
        DiscreteUpdater(pomdp),
        Dict{Symbol,Any}(:policy => "QMDP Optimal", :rule => "Bayes")) 
    for pomdp in [ 
        standard_reward_pomdp(10, balls_per_observation)
        for balls_per_observation in balls_per_observation_range
    ]
])

ibe_params = vec([
    (pomdp, 
        BeliefThresholdPolicy(pomdp, threshold, -1), 
        IBEUpdater(pomdp, PopperBonus(), 1.0),
        Dict{Symbol,Any}(:policy => (@sprintf "Threshold %s" threshold), :rule => "Popper")) 
    for pomdp in [ 
        standard_reward_pomdp(10, balls_per_observation)
        for balls_per_observation in balls_per_observation_range
    ], threshold in thresholds
])

ibe_optimal_params = vec([
    (pomdp, 
        solve(solver, pomdp),
        IBEUpdater(pomdp, PopperBonus(), 1.0),
        Dict{Symbol,Any}(:policy => "QMDP Optimal", :rule => "Popper")) 
    for pomdp in [ 
        standard_reward_pomdp(10, balls_per_observation)
        for balls_per_observation in balls_per_observation_range
    ]
])

params = vcat(bayes_params, ibe_params, ibe_optimal_params, bayes_optimal_params)
##
res = par_run_experiments(params, 10000)
##
name = @sprintf "threshold_comparison_%s" Dates.format(Dates.now(), "dd-mm-yyyy_HH-MM-SS")
Arrow.write(joinpath("results", "threshold_comparison.arrow"), res)
