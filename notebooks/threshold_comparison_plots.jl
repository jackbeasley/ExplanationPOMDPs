using Plots, StatsPlots, DataFrames
import Arrow
plotlyjs()
##
res = DataFrame(Arrow.Table("results/threshold_comparison.arrow"))
##
bayes_stats = combine(
    DataFrames.groupby(
        filter(row -> row.rule == "Bayes", res), 
        [:policy, :balls_per_observation]
    ),
     :r => mean,
)
##
bayes_fig = @df bayes_stats plot(:balls_per_observation, :r_mean, group=:policy,
    title="Reward vs. Draws for Bayes Agents",
    ylabel="Mean reward (n = 10000)", ylims=(-1.0, 1.0),
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
    ylabel="Mean reward (n = 10000)", ylims=(-1.0, 1.0),
    xlabel="Balls Drawn from Urn",
    legend=:bottomright,
    legendtitle="Policy",
    dpi=300,
)
png(popper_fig, "notebooks/popper_reward_draws.png")
popper_fig

