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
thresholds = collect(0.0:0.05:0.95)

balls_per_observation_range = [2^i for i in 1:6]
##
pomdps = [ 
    standard_reward_pomdp(10, balls_per_observation)
    for balls_per_observation in balls_per_observation_range
]
##
policies = [
        (
            threshold, 
            pomdp -> BeliefThresholdPolicy(pomdp, threshold, -1)
        )
        for threshold in thresholds
]
##
updaters = [
    ("Schupbach", pomdp -> IBEUpdater(pomdp, SchupbachBonus(), 1.0)),
    ("Good", pomdp -> IBEUpdater(pomdp, GoodBonus(), 1.0)),
    ("Popper", pomdp -> IBEUpdater(pomdp, PopperBonus(), 1.0)),
    ("Bayes", pomdp -> DiscreteUpdater(pomdp)),
]
##

params = vec([
    (pomdp, policy(pomdp), updater(pomdp), 
    Dict{Symbol,Any}(:policy => policy_name, :updater => updater_name))
 for pomdp in pomdps, (policy_name, policy) in policies, (updater_name, updater) in updaters])

##
res = par_run_experiments(params, 1000, true)
##
name = @sprintf "threshold_comparison_%s" Dates.format(Dates.now(), "dd-mm-yyyy_HH-MM-SS")
Arrow.write(joinpath("results", "threshold_comparison_new.arrow"), res)
