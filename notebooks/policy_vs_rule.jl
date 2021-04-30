using DataFrames, StatsBase, VegaLite
import Arrow
##
res = DataFrame(Arrow.Table("results/threshold_comparison_no_optim.arrow"))
##
mean_rewards = combine(
    groupby(res, [:policy, :updater, :balls_per_observation]), 
    :r => mean
)
##
p = filter(row -> row.balls_per_observation < 21, mean_rewards) |> @vlplot(
    title = "Average Reward vs. Belief Threshold by Update Rule ",
    mark = :line,
    x = {:policy, title = "Belief Threshold"},
    y = {:r_mean, title = "Mean Reward (n=10,000)"},
    color = {:updater, title = "Update Rule"},
    wrap = {:balls_per_observation, title = "Number of Draws from Urn", type = "ordinal"},
    height = 100,
    width = 220,
    columns = 2,
) # TODO: need to integrate policy / threshold
p |> save("policy_vs_rule.svg")
p
##
