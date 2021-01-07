---
title: "IBE-Based Discrete State Filters"
author: "Jack Beasley, `jbeasley@stanford.edu`"
date: "January 4th, 2020"
linestretch: 1
#geometry: "margin=1in"
papersize: letter
#classoption: twocolumn
bibliography: "./citations.bib"
#csl: "./apa.csl"
---

## Motivation

Proponents of inference to the best explanation (IBE) as an alternative to or
flavor of Bayesianism have proposed several different belief update rules
that differ significantly from the well-known Bayes rule. Proponents of these
rules argue they better capture human behavior
[@douvenProbabilisticAlternativesBayesianism2015] or are objectively superior
to Bayes rule under certain evaluation metrics, such as accuracy or speed of
convergence [@douvenInferenceBestExplanation2017]. However, proponents of
Bayesianism argue that Bayes rule is a uniquely optimal method of updating
one's credences using arguments such as the Dutch book argument (first
proposed by David Lewis [@lewisWhyConditionalize1999]). These arguments
construct a case where a player placing a wager will always lose unless they
are using Bayes rule to update their beliefs.

However, all this discussion about cases where IBE and Bayes rule are better
than one another raise the question: under what conditions and strategies
should we use one over the other? To answer this question, I propose using
these update rules in the context of a decision problem under uncertainty.
These sort of problems have been formalized as partially-observable Markov
decision processes (POMDPs) and discussed at length in the decision-making
literature where researchers have developed solvers that can find optimal
policies given a formal reward function. The key insight that this literature
brings to the IBE discussion is that reward can be formalized and might
differ significantly depending on domain. For example, a self-driving race
car on a closed track might care much more about speed than safety when
compared to one designed to be a taxi, despite very similar problems of image
recognition, localization, and driving. While getting a reward function right
can be very difficult, the POMDP formation allows us to suppose different
reward and world structures and see how different belief update rules perform
under different policies.

TODO: better POMDP intro

## Closing the Gap between POMDPs and Inference Research

In small, discrete cases, the belief update mechanisms used in the POMDP
literature aren't far off from direct application of Bayes rule, however,
there is still a small gap that must be closed to bring IBE rules into the
POMDP formulation. This section is aimed at unifying the language and
notation of the POMDP formulation with the update rule descriptions of the
IBE literature. Once this is complete, we can derive a variant of the
Bayesian POMDP belief update rule that replaces Bayes rule with an IBE measure.

First, I'll introduce PODMP concepts:

- A set of world states $S$
- A set of actions, $A$, an agent may take
- A set of observations, $O$, that the agent might encounter
- A transition function $T$ that dictates how states transition into one another
- A reward function that defines a scalar reward for the action an agent takes in a given state
- An observation function that defines which observations the agent gets given a state and action taken by the agent.

These components and concepts completely and unambiguously define the
predicament the agent is in and provide an unambiguous metric by which to
measure performance. While getting these definitions to match a real world
problem is difficult, we are supposing the problem so it isn't too difficult
to do this formally within this framework rather than informally in a paper.

In the IBE literature, constrained problems are often used to evaluate how a
belief update rule works in simulation, however, these are quite framed
within this framework because they don't explicitly model reward.

In an IBE formulation, states which an agent is uncertain about are referred
to as hypotheses and the observations are evidence.
