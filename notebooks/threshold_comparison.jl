include("experiment_base.jl")
using ExplanationPOMDPs.Beliefs
using IterTools
using Plots, StatsPlots
plotlyjs()
import DataFrames
import Arrow
##

# The goal of this notebook is to compare different selection thresholds at
# different difficulty levels
thresholds = [0.0, 0.1, 0.5, 0.9]

bayes_params = vec([
    (pomdp, 
        BeliefThresholdPolicy(pomdp, threshold, -1), 
        DiscreteUpdater(pomdp),
        Dict{Symbol,Any}(:threshold => threshold, :rule => "Bayes")) 
    for pomdp in [ 
        standard_reward_pomdp(10, balls_per_observation)
        for balls_per_observation in 5:25
    ], threshold in thresholds
])

ibe_params = vec([
    (pomdp, 
        BeliefThresholdPolicy(pomdp, threshold, -1), 
        IBEUpdater(pomdp, PopperBonus(), 1.0),
        Dict{Symbol,Any}(:threshold => threshold, :rule => "Popper")) 
    for pomdp in [ 
        standard_reward_pomdp(10, balls_per_observation)
        for balls_per_observation in 5:25
    ], threshold in thresholds
])
params = vcat(bayes_params, ibe_params)
##
res = run_experiments(params, 500)
##
bayes_stats = combine(
    DataFrames.groupby(
        filter(row -> row.rule == "Bayes", res), 
        [:threshold, :balls_per_observation]
    ),
     :r => mean
)
##
bayes_fig = @df bayes_stats plot(:balls_per_observation, :r_mean, group=:threshold,
    title="Reward vs. Draws for Bayes Agents",
    ylabel="Mean reward (n = 1000)", ylims=(-0.5, 0.5),
    xlabel="Balls Drawn from Urn"

)
bayes_fig
##
popper_stats = combine(
    DataFrames.groupby(
        filter(row -> row.rule == "Popper", res), 
        [:threshold, :balls_per_observation]
    ),
     :r => mean
)
##
popper_fig = @df popper_stats plot(:balls_per_observation, :r_mean, group=:threshold,
    title="Reward vs. Draws for IBE Agents",
    ylabel="Mean reward (n = 1000)", ylims=(-0.5, 0.5),
    xlabel="Balls Drawn from Urn"
)
popper_fig
##



Arrow.write("data.arrow", res)

