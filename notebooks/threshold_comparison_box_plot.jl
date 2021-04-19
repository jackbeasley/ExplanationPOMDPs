using Plots, StatsPlots, DataFrames, StatsBase, VegaLite
import Arrow
gr()
##
# res = DataFrame(Arrow.Table("results/threshold_comparison_new.arrow"))
##
reward_counts = combine(groupby(res, [:r, :policy, :updater, :balls_per_observation]), nrow => :count)
##
# limited = filter(row -> row.balls_per_observation < 26, reward_counts) 
##
reward_counts |> @vlplot(
    mark = :bar,
    x = {:r, type = "ordinal"},
    y = :count,
    column = :updater,
    row = :balls_per_observation
) # TODO: need to integrate policy / threshold


