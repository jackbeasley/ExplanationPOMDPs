---
title: "Explanations Meets Decision Theory"
author: "Jack Beasley"
---

*Note:* I changed topics late in the quarter after my proposal. My advisor
(Thomas Icard) and I found a new research direction that I noticed overlapped
well with this course, so I decided to get that project started with this
class project instead of going with my old power optimization project.


## Introduction

The goal of this paper is to start to bridge the gap between engineering
decision theory and theoretical work on explanations in epistemology. Given the
very different aims of these two communities, there is little to no
cross-pollination between the two, however, there are ample opportunities for
connections given that both center around inference and justified belief to
varying extents.

In this paper, I plan to first introduce the philosophical issues involved in
inference, critically including *abduction* or *Inference to the Best
Explanation* (IBE). Once these philosophical concepts are out of the way, I'll
introduce some of the simulation-based experiments philosophers have proposed
as evidence for inference by IBE or for inference according to Bayes rule.
I'll then frame one of these experiments in terms of a POMDP and show how
many of these differences in performance between these belief update rules
can be matched with optimal policies over bayesian update rules given
different reward structures. Essentially, I use the POMDP framing to make the
model more representative of real-world situations where incentives matter to
show how going with the most *explanatory* hypothesis as in IBE is better
thought of as a decision policy rather than a belief update rule.

## Background and Literature Review

### Inference in Philosophy

There are three distinct types of inference that often show up in the
philosophy of science and epistemology literature: deduction, abduction and
induction. To motivate these different forms of reasoning, I'll provide three
election-themed inferences that we can draw in each different form.

First, define a simple popular vote election between two candidates where the
candidates with more votes wins the election. Say we then know the following
premiss about candidate $A$ and candidate $B$:

1. The candidate with more votes wins the election.
2. Candidate $A$ won more votes than candidate $B$.

From these two facts, we can conclude *with certainty* that Candidate $A$ won
the election. This is the reasoning lets us derive truths through proof in
math, logic or any other axiomatizable system, however, it doesn't help us
much when we don't know the axioms for certain, which is common in the real
world. In philosophy, these sorts of inferences are lead to *a priori*
truths, or facts that don't require observation to know.

Thus, we turn to a famously shaky type of inference: induction. While
famously criticized by Hume, induction is the process of observing the world
and drawing inferences from prior observations. To see this form, and the
problems with it, consider a reliably Democratic precinct in an American
presidential election. Say we've observed this precinct vote for Democratic
candidates for the past twenty elections. By induction, we might conclude
that that same district will vote for the Democratic candidate in the next
election. However, this relies on the implicit assumption that the district
and the world at large doesn't change much. Usually this works out and in
practice this inference is often quite accurate. However, it is not
inconceivable that this inference would turn out to be wrong if people
changed their minds or the people in the district changed significantly. This
is a key difference with deduction as a correct deductive inference will
never be wrong.

Finally, we move to abduction, which is sort of like induction in reverse.
Where induction starts from a statistic or collection of observations and
makes an inference from that, abduction takes observations and infers a
statistic that might explain those observations best. Say you are a poll
worker for a single precinct in a city and are counting ballots. You might
count 80 votes for candidate $A$ and $5$ for candidate $B$. From these
observations, you might infer that candidate $A$ is getting the most votes in
that precinct's total vote because that would explain the skewed distribution
of votes that you observed. The key difference from induction is that we
appeal to our sense of what a good explanation is to make our inference for
abduction.

### Abduction, formally

Schumbach defintiions and experiments

### IBE vs. Bayes

Now that we understand what abduciton 

## IBE within a POMDP

Abductive tiger POMDP. There are tigers and monkeys in the room. Each 
Basically tiger POMDP...

### State space

### Action space

### Observation space

### Transition

### Rewards


## Results







