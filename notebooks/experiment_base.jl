using ExplanationPOMDPs.SingleObservationExplanation
using ExplanationPOMDPs.Policies
using POMDPs, BeliefUpdaters, POMDPPolicies, POMDPSimulators
using DataFrames
using ProgressMeter
using Distributed
##

standard_reward_pomdp(ballsInVase::Int, ballsDrawn::Int) = SingleObservationPOMDP(
    ballsInVase,     # Four balls in each vase
    ballsDrawn,     # Two balls drawn for each 
    0.0,   # No reward or penalty for omitting an answer
    1.0,   # 1 point reward for a correct guess
    -1.0,  # 1 point penalty for an incorrect guess
    1.0    # No discount
)

function run_experiment(problem::SingleObservationPOMDP, p::Policy, up::Updater, filter_init=true)::DataFrame
    hr = HistoryRecorder(max_steps=3)
    history = simulate(hr, problem, p, up, initialstate(problem))
    df = filter_init ? DataFrame(history[2:2]) : DataFrame(history)
    # df[:b] = [b.b for b in df[:b]]
    # df[:bp] = [bp.b for bp in df[:bp]]

    df[!,:s] = convert.(Int8, df[!,:s])
    df[!,:a] = convert.(Int8, df[!,:a])
    df[!,:sp] = convert.(Int8, df[!,:sp])
    df[!,:o] = convert.(Int8, df[!,:o])
    df[!,:t] = convert.(Int8, df[!,:t])
    df[!,:r] = convert.(Float16, df[!,:r])

    df[:balls_per_vase] = Int8(problem.balls_per_vase)
    df[:balls_per_observation] = Int8(problem.balls_per_observation)
    # df["r_correct"] = (Float16(v) for v in problem.r_correct)
    # df["r_incorrect"] = (Float16(v) for v in problem.r_incorrect)
    # df["r_no_choice"] = (Float16(v) for v in problem.r_no_choice)
    df[:discount] = Float16(problem.discount)
    empty_columns = [
        Symbol(names(df)[i])
        for (i, coltype) in enumerate(eltypes(df)) if coltype == Nothing
    ]
    select!(df, Not(empty_columns))
    select!(df, Not(:b))
    select!(df, Not(:bp))
    return df
end

function run_experiments(problem::POMDP, p::Policy, up::Updater, tags::Dict, n::Int, filter_init=true)::DataFrame
    history = run_experiment(problem, p, up, filter_init)
    history[:run] = Int16(1)
    if n > 1
        for i in 2:n
            run = run_experiment(problem, p, up, filter_init)
        run[:run] = Int16(i)
            append!(history, run)
        end
    end
        for (k, v) in tags
    history[k] = v
    end
    return history
end

const ExperimentParams = Tuple{SingleObservationPOMDP,BeliefThresholdPolicy{SingleObservationPOMDP},Updater,Dict{Symbol,Any}}

function run_experiments(params::AbstractVector{ExperimentParams}, n_per_param::Int, filter_init=true)::DataFrame
    return vcat([run_experiments(pomdp, p, up, tags, n_per_param, filter_init) for (pomdp, p, up, tags) in params]...)
end

function par_run_experiments(params::AbstractVector{ExperimentParams}, n_per_param::Int, filter_init=true)::DataFrame
    dfs = Vector{DataFrame}(undef, length(params))
    prog = Progress(length(params))
    Threads.@threads for i in 1:length(params)
        (pomdp, p, up, tags) = params[i]
        dfs[i] = run_experiments(pomdp, p, up, tags, n_per_param, filter_init)
        next!(prog)
    end

    return vcat(dfs...)
end

function dist_run_experiments(params::AbstractVector{ExperimentParams}, n_per_param::Int, filter_init=true)::DataFrame
    dfs = Vector{DataFrame}(undef, length(params))
    df = @showprogress @distributed (vcat) for param in params
        (pomdp, p, up, tags) = param
        run_experiments(pomdp, p, up, tags, n_per_param, filter_init)
    end
    return df
end
##


