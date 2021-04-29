module SingleObservationExplanation

using POMDPs, POMDPModelTools, Distributions, Printf, BeliefUpdaters
##
struct OneShotState{T <: Integer}
    end_state::Bool
    hypothesis_num::T
    step_num::T
end
export OneShotState

explanation_states(balls_in_vase::Int, observation_steps::Int) = Base.Iterators.flatten(
    (
        (OneShotState(true, -1, -1),),
        Base.Iterators.map(
            nums -> OneShotState(false, nums[1], nums[2]),
            Base.Iterators.product(0:balls_in_vase, 1:observation_steps)
        )
    )
)
export explanation_states

##

const Action = T where T <: Integer;
const Observation = T where T <: Integer;
const NUM_STEPS = 2
struct SingleObservationPOMDP <: POMDP{OneShotState,Action,Observation}
    balls_in_vase::Int
    balls_per_observation::Int
    r_no_choice::Float64
    r_correct::Float64
    r_incorrect::Float64
    discount::Float64
end
export SingleObservationPOMDP

POMDPs.discount(m::SingleObservationPOMDP) = m.discount

POMDPs.isterminal(::SingleObservationPOMDP, s::OneShotState) = s.end_state

const END_STATE = -1

# The state is equivalent to the configuration of balls in the vase. If the
# state is 5, then 5 balls are one color and n - 5 are the other.
POMDPs.states(pomdp::SingleObservationPOMDP) = collect(explanation_states(pomdp.balls_in_vase, NUM_STEPS))
function POMDPs.stateindex(pomdp::SingleObservationPOMDP, s::OneShotState)
    if s.end_state
        return 1
    end
    
    return ((pomdp.balls_in_vase + 1) * (s.step_num - 1) + s.hypothesis_num) + 2
end
POMDPs.initialstate(pomdp::SingleObservationPOMDP) = SparseCat(
    states(pomdp),
    start_state_probs(pomdp)
)


# Each observation is the cumulative affect of observing balls_per_observation
# balls, so (1, balls_per_observation) represent all possible observations
# because we assume that order shouldn't affect belief in the one-shot context.
# This should be reasonable because
POMDPs.observations(pomdp::SingleObservationPOMDP) = collect(0:pomdp.balls_per_observation)
POMDPs.obsindex(::SingleObservationPOMDP, o::Observation) = o + 1
# Next state will be the one with 
function POMDPs.observation(pomdp::SingleObservationPOMDP, s::OneShotState) 
    if s.end_state
        # Transition probabilities make this impossible, so this uniform choice is entirely arbitrary
        return DiscreteUniform(0, pomdp.balls_per_observation)
    end
    return Binomial(
        pomdp.balls_per_observation, 
        s.hypothesis_num / pomdp.balls_in_vase
    )
end

# Only actions are suspending judgment (a = -1) or choosing a vase (0 <= a <= n)
# configuration
POMDPs.actions(pomdp::SingleObservationPOMDP) = collect(-1:pomdp.balls_in_vase)
POMDPs.actionindex(::SingleObservationPOMDP, a::Action) = a + 2

initial_belief(pomdp::SingleObservationPOMDP) = DiscreteBelief(
    pomdp,
    start_state_probs(pomdp)
)
export initial_belief

function POMDPs.reward(pomdp::SingleObservationPOMDP, s::OneShotState, a::Action)
    if isterminal(pomdp, s)
        return 0
    end

    if a == -1
        return pomdp.r_no_choice
    elseif a == s.hypothesis_num
        return pomdp.r_correct
    else
        return pomdp.r_incorrect
    end
end

# Uniform probability to all hypothesis states, which each correspond to vase
# configurations, with no chance of going to end state
function start_state_probs(pomdp::SingleObservationPOMDP)::Vector{Float64}
    num_vase_configurations = pomdp.balls_in_vase + 1
    prob = 1.0 / num_vase_configurations
    return [s.step_num == 1 ? prob : 0.0 for s in states(pomdp)]
end
export start_state_probs

function POMDPs.transition(pomdp::SingleObservationPOMDP, s::OneShotState, a::Action)
    n_states = length(POMDPs.states(pomdp))
    if s.step_num < NUM_STEPS
        # Go to the next hypothesis state
        next_state_ind = stateindex(pomdp, OneShotState(false, s.hypothesis_num, s.step_num + 1))

        return SparseCat(
            states(pomdp),
            [i == next_state_ind ? 1.0 : 0.0 for i in 1:n_states]
        )
    end
    # Goto end state
    return SparseCat(
        states(pomdp),
        vcat([1.0], zeros(n_states - 1))
    ) 
end

end
