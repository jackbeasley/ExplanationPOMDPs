using Plots, StatsPlots, DataFrames, StatsBase
import Arrow
gr()
##
res = DataFrame(Arrow.Table("results/threshold_comparison_new.arrow"))
##
average_rewards_for_rule(df::DataFrame, updater::String) = combine(
    groupby(
        combine(
            groupby(
                filter(row -> row.updater == updater, df), 
                [:policy, :balls_per_observation, :s]
            ),
            :r => mean => :r_state_mean,
        ),
        [:policy, :balls_per_observation]
    ), 
    :r_state_mean => mean => :r_mean
)
##
bayes_stats = average_rewards_for_rule(res, "Bayes")
##
bayes_fig = @df bayes_stats plot(:balls_per_observation, :r_mean, group=:policy,
    title="Reward vs. Draws for Bayes Agents",
    ylabel="Mean reward (n = 10000)", ylims=(-1.0, 1.0),
    xlabel="Balls Drawn from Urn",
    legend=:bottomright,
    legendtitle="Policy",
    dpi=500,
)
png(bayes_fig, "notebooks/bayes_reward_draws.png")
bayes_fig
##
popper_stats = average_rewards_for_rule(res, "Popper")
##
popper_fig = @df popper_stats plot(:balls_per_observation, :r_mean, group=:policy,
    title="Reward vs. Draws for Popper Rule Agents",
    ylabel="Mean reward (n = 10000)", ylims=(-1.0, 1.0),
    xlabel="Balls Drawn from Urn",
    legend=:bottomright,
    legendtitle="Policy",
    dpi=500,
)
png(popper_fig, "notebooks/popper_reward_draws.png")
popper_fig
## GOod
good_stats = average_rewards_for_rule(res, "Good")
##
good_fig = @df good_stats plot(:balls_per_observation, :r_mean, group=:policy,
    title="Reward vs. Draws for Good Rule Agents",
    ylabel="Mean reward (n = 10000)", ylims=(-1.0, 1.0),
    xlabel="Balls Drawn from Urn",
    legend=:bottomright,
    legendtitle="Policy",
    dpi=500,
)
png(good_fig, "notebooks/good_reward_draws.png")
good_fig

## Schupbach
schupbach_stats = average_rewards_for_rule(res, "Schupbach")
##
schupbach_fig = @df schupbach_stats plot(:balls_per_observation, :r_mean, group=:policy,
    title="Reward vs. Draws for Schupbach Rule Agents",
    ylabel="Mean reward (n = 10000)", ylims=(-1.0, 1.0),
    xlabel="Balls Drawn from Urn",
    legend=:bottomright,
    legendtitle="Policy",
    dpi=500,
)
png(schupbach_fig, "notebooks/schupbach_reward_draws.png")
schupbach_fig


