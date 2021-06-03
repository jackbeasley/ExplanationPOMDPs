using VegaLite, DataFrames, StatsBase
import Arrow
##
res = DataFrame(Arrow.Table("results/threshold_comparison.arrow"))
res[!, :policy] = string.(res.policy)
##
average_rewards = combine(
    groupby(
        combine(
            groupby(
                res,
                [:policy, :balls_per_observation, :s, :updater]
            ),
            :r => mean => :r_state_mean,
        ),
        [:policy, :balls_per_observation, :updater]
    ), 
    :r_state_mean => mean => :r_mean
)

##
p = average_rewards |> @vlplot(
    title = "Reward vs. Draws for Bayes Agents",
    mark = :line,
    x = {:balls_per_observation, title = "Balls Drawn from Urn"},
    y = {:r_mean, title = "Mean reward (n = 10000)"},
    color = {:policy, title = "Policy Threshold"},
    wrap = {:updater, title = "Belief Update Rule"},
    columns = 2,
)
##
p |> save("reward_draws.pdf")
##



