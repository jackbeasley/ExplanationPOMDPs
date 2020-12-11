module Beliefs

using POMDPs, BeliefUpdaters

const BayesUpdater = DiscreteUpdater

"""
    IBEUpdater
An updater type to update discrete belief using an explanationist rule
# Constructor
    IBEUpdater(pomdp::POMDP)
# Fields
- `pomdp <: POMDP`
"""
mutable struct IBEUpdater{P <: POMDP} <: Updater
    pomdp::P
end

uniform_belief(up::IBEUpdater) = uniform_belief(up.pomdp)

function initialize_belief(bu::IBEUpdater, dist::Any)
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

function update(bu::IBEUpdater, b::DiscreteBelief, a, o)
    pomdp = bu.pomdp
    state_space = b.state_list
    belief_probs = zeros(length(state_space))

    for (s_ind, s) in enumerate(state_space)

        if pdf(b, s) > 0.0
            td = transition(pomdp, s, a)

            for (sp, tp) in weighted_iterator(td)
                spi = stateindex(pomdp, sp)
                op = obs_weight(pomdp, s, a, sp, o) # shortcut for observation probability from POMDPModelTools

                belief_probs[spi] += op * tp * b.b[s_ind]
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

update(bu::IBEUpdater, b::Any, a, o) = update(bu, initialize_belief(bu, b), a, o)



end
