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
end
export SingleObservationPOMDP

POMDPs.discount(m::SingleObservationPOMDP) = m.discount

POMDPs.isterminal(pomdp::SingleObservationPOMDP, s::OneShotState) = s > pomdp.balls_per_vase
ball_count(pomdp::SingleObservationPOMDP, s::OneShotState) = s <= pomdp.balls_per_vase ? s : s - pomdp.balls_per_vase - 1
export ball_count

# The state is equivalent to the configuration of balls in the vase. If the
# state is 5, then 5 balls are one color and n - 5 are the other.
POMDPs.states(pomdp::SingleObservationPOMDP) = collect(0:(2 * pomdp.balls_per_vase) + 1)
POMDPs.stateindex(::SingleObservationPOMDP, s::OneShotState) = s + 1
# Assigns an uniform probability to starting in any non-terminal state / vase
# configuration
function POMDPs.initialstate(pomdp::SingleObservationPOMDP) 
    number_vase_configurations = pomdp.balls_per_vase + 1
    uniform_prob = 1.0 / number_vase_configurations
    return SparseCat(
    states(pomdp),
    vcat(fill(uniform_prob, number_vase_configurations), zeros(number_vase_configurations)),
)
end


# Each observation is the cumulative affect of observing balls_per_observation
# balls, so (1, balls_per_observation) represent all possible observations
# because we assume that order shouldn't affect belief in the one-shot context.
# This should be reasonable because
POMDPs.observations(pomdp::SingleObservationPOMDP) = collect(1:pomdp.balls_per_observation)
POMDPs.obsindex(::SingleObservationPOMDP, o::Observation) = o
# Assigns an uniform probability to each observation for the first and only
# observation
POMDPs.initialobs(pomdp::SingleObservationPOMDP, s::OneShotState) = Binomial(
    pomdp.balls_per_observation, 
    s / length(states(pomdp))
)
POMDPs.observation(pomdp::SingleObservationPOMDP, s::OneShotState, a::Action) = initialobs(pomdp, s)

# Only actions are suspending judgment (a = 0) or choosing a vase (1 < a < n)
# configuration
POMDPs.actions(pomdp::SingleObservationPOMDP) = collect(0:pomdp.balls_per_vase)
POMDPs.actionindex(::SingleObservationPOMDP, a::Action) = a + 1

initial_belief(pomdp::SingleObservationPOMDP) = DiscreteBelief(pomdp.n_balls)
export initial_belief

function POMDPs.reward(pomdp::SingleObservationPOMDP, s::OneShotState, a::Action)
    if a == 0
        return pomdp.r_no_choice
    elseif a == s
        return pomdp.r_correct
    else
        return pomdp.r_incorrect
    end
end

POMDPs.transition(pomdp::SingleObservationPOMDP, s::OneShotState, a::Action) = initialstate(pomdp)

end
