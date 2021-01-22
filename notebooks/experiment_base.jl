using ExplanationPOMDPs.SingleObservationExplanation
using ExplanationPOMDPs.Policies
using POMDPs, BeliefUpdaters, POMDPPolicies, POMDPSimulators
using DataFrames
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
    df[:b] = [b.b for b in df[:b]]
    df[:bp] = [bp.b for bp in df[:bp]]

    df["balls_per_vase"] = problem.balls_per_vase
    df["balls_per_observation"] = problem.balls_per_observation
    df["r_correct"] = problem.r_correct
    df["r_incorrect"] = problem.r_incorrect
    df["r_no_choice"] = problem.r_no_choice
    df["discount"] = problem.discount
    empty_columns = [
        Symbol(names(df)[i])
        for (i, coltype) in enumerate(eltypes(df)) if coltype == Nothing
    ]
    select!(df, Not(empty_columns))
    return df
end

function run_experiments(problem::POMDP, p::Policy, up::Updater, tags::Dict, n::Int, filter_init=true)::DataFrame
    history = run_experiment(problem, p, up, filter_init)
    history[:run] = 1
    if n > 1
        for i in 2:n
            run = run_experiment(problem, p, up, filter_init)
            run[:run] = i
            append!(history, run)
        end
    end
    for (k, v) in tags
        history[k] = v
    end
    return history
end

const ExperimentParams = Tuple{SingleObservationPOMDP,Policy,Updater,Dict{Symbol,Any}}

function run_experiments(params::Vector{ExperimentParams}, n_per_param::Int, filter_init=true)::DataFrame
    return vcat([run_experiments(pomdp, p, up, tags, n_per_param, filter_init) for (pomdp, p, up, tags) in params]...)
end

function par_run_experiments(params::Vector{ExperimentParams}, n_per_param::Int, filter_init=true)::DataFrame
    chan = Channel{DataFrame}(length(params))
    for (pomdp, p, up, tags) in params
        Threads.@spawn put!(chan, run_experiments($pomdp, $p, $up, $tags, $n_per_param, $filter_init))
    end
    df = take!(chan)
    num_finished = 1
    while num_finished < length(params)
        next_df = take!(chan)
        num_finished += 1
        vcat(df, next_df)
    end
    return df
end
##


