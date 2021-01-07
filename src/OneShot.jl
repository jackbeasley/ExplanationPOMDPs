module SingleObservationExplanation

using POMDPs, POMDPModelTools, Distributions, Printf, BeliefUpdaters


const OneShotState = T where T <: Integer;
const Action = T where T <: Integer;
const Observation = T where T <: Integer;

struct SingleObservationPOMDP <: POMDP{OneShotState,Action,Observation}
    balls_per_vase::Int
    balls_per_observation::Int
    r_no_choice::Float64
    r_correct::Float64
    r_incorrect::Float64
    discount::Float64
    initial_belief::B
end
export SingleObservationPOMDP

POMDPs.discount(m::SingleObservationPOMDP) = m.discount

POMDPs.isterminal(::SingleObservationPOMDP, s::OneShotState) = s == -1

const START_STATE = -2
const END_STATE = -1

# The state is equivalent to the configuration of balls in the vase. If the
# state is 5, then 5 balls are one color and n - 5 are the other.
POMDPs.states(pomdp::SingleObservationPOMDP) = collect(-2:pomdp.balls_per_vase)
POMDPs.stateindex(::SingleObservationPOMDP, s::OneShotState) = s + 3
POMDPs.initialstate(pomdp::SingleObservationPOMDP) = SparseCat(
    states(pomdp),
    vcat([1.0], zeros(pomdp.balls_per_vase + 2))
)


# Each observation is the cumulative affect of observing balls_per_observation
# balls, so (1, balls_per_observation) represent all possible observations
# because we assume that order shouldn't affect belief in the one-shot context.
# This should be reasonable because
POMDPs.observations(pomdp::SingleObservationPOMDP) = collect(0:pomdp.balls_per_observation)
POMDPs.obsindex(::SingleObservationPOMDP, o::Observation) = o + 1
# Next state will be the one with 
function POMDPs.observation(pomdp::SingleObservationPOMDP, s::OneShotState, a::Action, sn::OneShotState) 
    if sn < 0
        # Transition probabilities make this impossible, so this uniform choice is entirely arbitrary
        return DiscreteUniform(0, pomdp.balls_per_observation)
    end
    return Binomial(
        pomdp.balls_per_observation, 
        sn / pomdp.balls_per_vase
    )
end

# Only actions are suspending judgment (a = -1) or choosing a vase (0 <= a <= n)
# configuration
POMDPs.actions(pomdp::SingleObservationPOMDP) = collect(-1:pomdp.balls_per_vase)
POMDPs.actionindex(::SingleObservationPOMDP, a::Action) = a + 2

initial_belief(pomdp::SingleObservationPOMDP) = DiscreteBelief(
    pomdp,
    uniform_state_probs(pomdp)
)
export initial_belief

function POMDPs.reward(pomdp::SingleObservationPOMDP, s::OneShotState, a::Action)
    if isterminal(pomdp, s)
        return 0
    end

    if a == -1
        return pomdp.r_no_choice
    elseif a == s
        return pomdp.r_correct
    else
        return pomdp.r_incorrect
    end
end

# Uniform probability to all terminal states, which each correspond to vase
# configurations, with no chance ot 
function uniform_state_probs(pomdp::SingleObservationPOMDP)::Vector{Float64}
    num_vase_configurations = pomdp.balls_per_vase + 1
    prob = 1.0 / num_vase_configurations
    return vcat([0.0, 0.0], fill(prob, num_vase_configurations))
end
export uniform_state_probs

function POMDPs.transition(pomdp::SingleObservationPOMDP, s::OneShotState, a::Action)
    if s == START_STATE
        return SparseCat(
            states(pomdp),
            uniform_state_probs(pomdp)
        )
    end
    # Goto end state
    return SparseCat(
        states(pomdp),
        vcat([0.0, 1.0], zeros(pomdp.balls_per_vase + 1))
    ) 
end

end
