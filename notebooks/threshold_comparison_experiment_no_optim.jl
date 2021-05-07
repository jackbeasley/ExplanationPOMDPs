include("experiment_base.jl")
using ExplanationPOMDPs.Urn
using ExplanationPOMDPs.Beliefs
using Printf, DataFrames
import Arrow
##

# The goal of this notebook is to compare different selection thresholds at
# different difficulty levels
thresholds = collect(0.0:0.01:0.95)

balls_per_observation_range = [5, 10, 15, 20, 25, 30]
pomdps = [ 
    standard_reward_pomdp(10, balls_per_observation)
    for balls_per_observation in balls_per_observation_range
]
policies = [
        (
            threshold, 
            pomdp -> BeliefThresholdPolicy(pomdp, threshold, actions(pomdp)[1])
        )
        for threshold in thresholds
]
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
let res
    res = par_run_experiments(params, 10000, true)
    Arrow.write(joinpath("results", "threshold_comparison_no_optim.arrow"), res, compress=:zstd)
end
##
#    name = @sprintf "threshold_comparison_%s" Dates.format(Dates.now(), "dd-mm-yyyy_HH-MM-SS")
#    Arrow.write(joinpath("results", "threshold_comparison_no_optim.arrow"), res)
