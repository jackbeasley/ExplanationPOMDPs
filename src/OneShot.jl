module SingleObservationExplanation

using POMDPs, POMDPModelTools, Distributions, Printf

const State = T where T <: Integer;
const Action = T where T <: Integer;
const Observation = T where T <: Integer;

struct SingleObservationPOMDP <: POMDP{State,Action,Observation}
    balls_per_vase::Int
    balls_per_observation::Int
    r_no_choice::Float64
    r_correct::Float64
    r_incorrect::Float64
    discount::Float64
end
export SingleObservationPOMDP

POMDPs.discount(m::SingleObservationPOMDP) = m.discount

# The state is equivalent to the configuration of balls in the vase. If the
# state is 5, then 5 balls are one color and n - 5 are the other.
POMDPs.states(pomdp::SingleObservationPOMDP) = collect(1:pomdp.balls_per_vase)
POMDPs.stateindex(::SingleObservationPOMDP, s::State) = s
# Assigns an uniform probability to starting in any given state / vase
# configuration
POMDPs.initialstate(pomdp::SingleObservationPOMDP) = SparseCat(
    states(pomdp), 
    fill(1.0 / length(states(pomdp)), length(states(pomdp))),
)

# Each observation is the cumulative affect of observing balls_per_observation
# balls, so (1, balls_per_observation) represent all possible observations
# because we assume that order shouldn't affect belief in the one-shot context.
# This should be reasonable because
POMDPs.observations(pomdp::SingleObservationPOMDP) = collect(1:pomdp.balls_per_observation)
POMDPs.obsindex(::SingleObservationPOMDP, o::Observation) = o
# Assigns an uniform probability to each observation for the first and only
# observation
POMDPs.initialobs(pomdp::SingleObservationPOMDP, s::State) = Binomial(
    pomdp.balls_per_observation, 
    s / length(states(pomdp))
)
POMDPs.observation(pomdp::SingleObservationPOMDP, s::State, a::Action) = initialobs(pomdp, s)

# Only actions are suspending judgment (a = 0) or choosing a vase (1 < a < n)
# configuration
POMDPs.actions(pomdp::SingleObservationPOMDP) = collect(0:pomdp.balls_per_vase)
POMDPs.actionindex(::SingleObservationPOMDP, a::Action) = a + 1

function POMDPs.reward(pomdp::SingleObservationPOMDP, s::State, a::Action)
    if a == 0
        return pomdp.r_no_choice
    elseif a == s
        return pomdp.r_correct
    else
        return pomdp.r_incorrect
    end
end

POMDPs.transition(pomdp::SingleObservationPOMDP, s::State, a::Action) = initialstate(pomdp)

end
