include("experiment_base.jl")
using ExplanationPOMDPs.Beliefs
using Printf
using SARSOP
import DataFrames
import Arrow
import Dates
##

# The goal of this notebook is to compare different selection thresholds at
# different difficulty levels
thresholds = [0.0, 0.3, 0.6, 0.9]

balls_per_observation_range = 5:10:105

pomdps = [ 
    standard_reward_pomdp(10, balls_per_observation)
    for balls_per_observation in balls_per_observation_range
]
solver = SARSOPSolver(precision=1.0e-7) 

policies = vcat([
        (
            (@sprintf "Threshold %s" threshold), 
            pomdp -> BeliefThresholdPolicy(pomdp, threshold, -1)
        )
        for threshold in thresholds
], [("Optimal", pomdp -> solve(solver, pomdp))])
updaters = [
    ("Schupbach", pomdp -> IBEUpdater(pomdp, SchupbachBonus(), 1.0)),
    ("Good", pomdp -> IBEUpdater(pomdp, GoodBonus(), 1.0)),
    ("Popper", pomdp -> IBEUpdater(pomdp, PopperBonus(), 1.0)),
    ("Bayes", pomdp -> DiscreteUpdater(pomdp)),
]

params = vec([
    (pomdp, policy(pomdp), updater(pomdp), 
    Dict{Symbol,Any}(:policy => policy_name, :updater => updater_name))
 for pomdp in pomdps, (policy_name, policy) in policies, (updater_name, updater) in updaters])

##
res = par_run_experiments(params, 10000, true)
##
Arrow.write(joinpath("results", "threshold_comparison_new.arrow"), res)
