include("experiment_base.jl")
using ExplanationPOMDPs.Beliefs
using Plots, StatsPlots
using Printf
using QMDP
plotlyjs()
import DataFrames
import Arrow
##

# The goal of this notebook is to compare different selection thresholds at
# different difficulty levels
thresholds = [0.0, 0.3, 0.6, 0.9]

solver = QMDPSolver(max_iterations=20,
                    belres=1e-3,
                    verbose=true
                   ) 

bayes_params = vec([
    (pomdp, 
        BeliefThresholdPolicy(pomdp, threshold, -1), 
        DiscreteUpdater(pomdp),
        Dict{Symbol,Any}(:policy => (@sprintf "Threshold %s" threshold), :rule => "Bayes")) 
    for pomdp in [ 
        standard_reward_pomdp(10, balls_per_observation)
        for balls_per_observation in 5:25
    ], threshold in thresholds
])

bayes_optimal_params = vec([
    (pomdp, 
        solve(solver, pomdp),
        DiscreteUpdater(pomdp),
        Dict{Symbol,Any}(:policy => "QMDP Optimal", :rule => "Bayes")) 
    for pomdp in [ 
        standard_reward_pomdp(10, balls_per_observation)
        for balls_per_observation in 5:25
    ]
])

ibe_params = vec([
    (pomdp, 
        BeliefThresholdPolicy(pomdp, threshold, -1), 
        IBEUpdater(pomdp, PopperBonus(), 1.0),
        Dict{Symbol,Any}(:policy => (@sprintf "Threshold %s" threshold), :rule => "Popper")) 
    for pomdp in [ 
        standard_reward_pomdp(10, balls_per_observation)
        for balls_per_observation in 5:25
    ], threshold in thresholds
])

ibe_optimal_params = vec([
    (pomdp, 
        solve(solver, pomdp),
        IBEUpdater(pomdp, PopperBonus(), 1.0),
        Dict{Symbol,Any}(:policy => "QMDP Optimal", :rule => "Popper")) 
    for pomdp in [ 
        standard_reward_pomdp(10, balls_per_observation)
        for balls_per_observation in 5:25
    ]
])

params = vcat(bayes_params, ibe_params, ibe_optimal_params, bayes_optimal_params)
##
res = run_experiments(params, 1000)
##
bayes_stats = combine(
    DataFrames.groupby(
        filter(row -> row.rule == "Bayes", res), 
        [:policy, :balls_per_observation]
    ),
     :r => mean
)
##
bayes_fig = @df bayes_stats plot(:balls_per_observation, :r_mean, group=:policy,
    title="Reward vs. Draws for Bayes Agents",
    ylabel="Mean reward (n = 1000)", ylims=(-0.5, 0.5),
    xlabel="Balls Drawn from Urn",
    legend=:bottomright,
    legendtitle="Policy",
    dpi=300,
)
png(bayes_fig, "notebooks/bayes_reward_draws.png")
bayes_fig
##
popper_stats = combine(
    DataFrames.groupby(
        filter(row -> row.rule == "Popper", res), 
        [:policy, :balls_per_observation]
    ),
     :r => mean
)
##
popper_fig = @df popper_stats plot(:balls_per_observation, :r_mean, group=:policy,
    title="Reward vs. Draws for IBE Agents",
    ylabel="Mean reward (n = 1000)", ylims=(-0.5, 0.5),
    xlabel="Balls Drawn from Urn",
    legend=:bottomright,
    legendtitle="Policy",
    dpi=300,
)
png(popper_fig, "notebooks/popper_reward_draws.png")
popper_fig

