module Beliefs

using POMDPs, BeliefUpdaters, POMDPModelTools

const BayesUpdater = DiscreteUpdater

abstract type IBEBonusRule end
export IBEBonusRule

struct PopperBonus <: IBEBonusRule end

function bonus(::PopperBonus, 
    prior_p_state::T, 
    p_observation_given_state::T, 
    prior_p_observation::T)::T where {T <: Real}
    return (p_observation_given_state - prior_p_observation) / (p_observation_given_state + prior_p_observation)
end
export PopperBonus, bonus

struct GoodBonus <: IBEBonusRule end
function rescale(x, a=2)
	d = 2 * a^2
    	if x >= 0
		return 1 - exp(-1 * (x^2) / d)
	else
		return -1 + exp(-1 * (x^2) / d)
	end
end
function bonus(::GoodBonus, 
    prior_p_state::T, 
    p_observation_given_state::T, 
    prior_p_observation::T)::T where {T <: Real}
    val = @fastmath log(p_observation_given_state / prior_p_observation) 
    return rescale(val)
end
export GoodBonus, bonus

struct SchupbachBonus <: IBEBonusRule end
function bonus(::SchupbachBonus, 
    prior_p_state::T,
    p_observation_given_state::T, 
    prior_p_observation::T)::T where {T <: Real}
    p_not_observation_given_state = 1.0 - p_observation_given_state
    p_state_given_not_observation = prior_p_state * p_not_observation_given_state

    numerator = p_observation_given_state - p_state_given_not_observation
    denominator = p_observation_given_state + p_state_given_not_observation

    if numerator == 0.0 && denominator == 0.0
        return -1.0
    end
    res = numerator / denominator
    return res
end
export SchupbachBonus, bonus

"""
    IBEUpdater
An updater type to update discrete belief using an explanationist rule
# Constructor
    IBEUpdater(pomdp::POMDP, bonusRule::R, bonusWeight)
# Fields
- `pomdp <: POMDP`
- `bonusRule <: IBEBonusRule`
- `bonusWeight <: Real`
"""
mutable struct IBEUpdater{P <: POMDP,R <: IBEBonusRule} <: Updater
    pomdp::P
    bonusRule::R
    bonusWeight::Real
end
export IBEUpdater

uniform_belief(up::IBEUpdater) = uniform_belief(up.pomdp)

function POMDPs.initialize_belief(bu::IBEUpdater, dist::Any)
    state_list = ordered_states(bu.pomdp)
    ns = length(state_list)
    b = zeros(ns)
    belief = DiscreteBelief(bu.pomdp, state_list, b)
    for s in support(dist)
        sidx = stateindex(bu.pomdp, s)
        belief.b[sidx] = pdf(dist, s)
    end
    return belief
end

function POMDPs.update(bu::IBEUpdater, b::DiscreteBelief, a, o)
    pomdp = bu.pomdp
    state_space = b.state_list
    belief_probs = zeros(length(state_space))

    # Note: belief updates use POMDP terminology, rather than Inference-to-best
    # explanation terminology. Thus, hypotheses are states (unknown which is
    # right to the agent in a POMDP) and evidence (observed by the agent) is an
    # observation.

    # Calculate P(Observation | State)P(State) + (Scaled Bonus) for each State
    for (s_ind, s) in enumerate(state_space)

        # If state is impossible given current beliefs the update will always
        # result in zero
        if pdf(b, s) > 0.0

            # The agent is assumed to know the structure of the problem, so use
            # probabilities of each next state from our current state and
            # action. In the OneShot context, this is equivalent to knowing
            # that there is equal chance of any hypothesis / urn configuration
            # being selected by the experimenters. In other words, this is a
            # uniform prior over hypotheses / states, but specified by the
            # agent's knowledge of the problem.
            td = transition(pomdp, s, a)
            
            # Overall probability of observation no matter what the next state
            # is given current state belief (pdf(b, s)) and transition
            # probabilities
            p_observation = sum(
                pdf(b, s) * prob * pdf(observation(pomdp, s, a, state), o) 
                for (state, prob) in weighted_iterator(td)
            )

            # Loop through every possible next state. Note that in a one-shot
            # context where all evidence is given in single observation, there
            # is only one final state where the transition probability is 1.0,
            # so this loop will, in practice, only add a non-zero value to the
            # belief for that final state.
            for (sp, tp) in weighted_iterator(td)
                spi = stateindex(pomdp, sp)

                # Calculate the objective probability of this observation given
                # the state we are calculating beliefs for, s, and for each
                # possible next state.
                p_observation_given_state = pdf(observation(pomdp, s, a, sp), o)

                prior_p_state = tp * pdf(b, s)

                bayes_p_state_given_observation = prior_p_state * p_observation_given_state

                belief_probs[spi] += bayes_p_state_given_observation
                if bayes_p_state_given_observation > 0.0
                    explanatory_bonus = bu.bonusWeight * bayes_p_state_given_observation * bonus(
                        bu.bonusRule, 
                        prior_p_state, 
                        bayes_p_state_given_observation, 
                        p_observation
                    )
                    belief_probs[spi] += explanatory_bonus
                end
        end
    end
    end

    bp_sum = sum(belief_probs)

    if bp_sum == 0.0
        error("""
          Failed discrete belief update: new probabilities sum to zero.
          b = $b
          a = $a
          o = $o
          Failed discrete belief update: new probabilities sum to zero.
          """)
    end

# Normalize
    belief_probs ./= bp_sum

    return DiscreteBelief(pomdp, b.state_list, belief_probs)
end

POMDPs.update(bu::IBEUpdater, b::Any, a, o) = update(bu, initialize_belief(bu, b), a, o)

end
