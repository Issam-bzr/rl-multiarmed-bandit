# multi_armed_bandit.py
# Author: BOUTEROUATA Issam Salah Eddine
# Description: A simulation of the Multi-Armed Bandit (MAB) problem —
# one of the foundational problems in Reinforcement Learning.
#
# The "bandit" is a slot machine with N arms. Each arm pays out a reward
# drawn from a different probability distribution. The agent doesn't know
# which arm is best — it has to learn through experience.
#
# This script implements and COMPARES three classic strategies:
#   1. Epsilon-Greedy      — explore randomly with probability epsilon
#   2. Upper Confidence Bound (UCB) — prefer arms that haven't been tried much
#   3. Random baseline     — pure random selection (no learning)
#
# Why this matters for AI quality:
# MAB is the mathematical foundation behind A/B testing, recommendation
# systems, and adaptive personalization — exactly the kind of system
# an AI Quality Analyst evaluates in production.

import random
import math

# ─────────────────────────────────────────
# BANDIT SETUP
# ─────────────────────────────────────────
NUM_ARMS = 5         # number of slot machine arms
NUM_STEPS = 1000     # how many pulls per run
NUM_RUNS = 50        # average over this many independent simulations
EPSILON = 0.1        # exploration rate for epsilon-greedy

# True reward means for each arm (agent does NOT know these)
# Arm 3 (index 2) is the true best arm with mean reward of 0.8
TRUE_MEANS = [0.2, 0.4, 0.8, 0.3, 0.6]


def pull_arm(arm_index):
    """Simulate pulling an arm — returns a reward sampled from a Bernoulli distribution."""
    return 1 if random.random() < TRUE_MEANS[arm_index] else 0


# ─────────────────────────────────────────
# STRATEGY 1: Epsilon-Greedy
# ─────────────────────────────────────────
def run_epsilon_greedy(epsilon=EPSILON):
    counts  = [0] * NUM_ARMS      # how many times each arm was pulled
    values  = [0.0] * NUM_ARMS    # estimated reward for each arm
    total_reward = 0
    rewards = []

    for step in range(NUM_STEPS):
        # Explore with probability epsilon, else exploit best known arm
        if random.random() < epsilon:
            arm = random.randint(0, NUM_ARMS - 1)
        else:
            arm = values.index(max(values))

        reward = pull_arm(arm)
        counts[arm] += 1
        # Incremental mean update
        values[arm] += (reward - values[arm]) / counts[arm]
        total_reward += reward
        rewards.append(total_reward / (step + 1))  # running average

    return rewards, values


# ─────────────────────────────────────────
# STRATEGY 2: Upper Confidence Bound (UCB1)
# ─────────────────────────────────────────
def run_ucb():
    counts  = [0] * NUM_ARMS
    values  = [0.0] * NUM_ARMS
    total_reward = 0
    rewards = []

    for step in range(NUM_STEPS):
        # Pull each arm once to initialize
        if step < NUM_ARMS:
            arm = step
        else:
            # UCB score = estimated value + exploration bonus
            ucb_scores = [
                values[a] + math.sqrt(2 * math.log(step + 1) / counts[a])
                for a in range(NUM_ARMS)
            ]
            arm = ucb_scores.index(max(ucb_scores))

        reward = pull_arm(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        total_reward += reward
        rewards.append(total_reward / (step + 1))

    return rewards, values


# ─────────────────────────────────────────
# STRATEGY 3: Random Baseline
# ─────────────────────────────────────────
def run_random():
    total_reward = 0
    rewards = []
    for step in range(NUM_STEPS):
        arm = random.randint(0, NUM_ARMS - 1)
        reward = pull_arm(arm)
        total_reward += reward
        rewards.append(total_reward / (step + 1))
    return rewards


def average_runs(strategy_fn, runs=NUM_RUNS):
    """Run a strategy multiple times and average the reward curves."""
    all_rewards = []
    all_final_values = []
    for _ in range(runs):
        if strategy_fn == run_random:
            r = strategy_fn()
            all_rewards.append(r)
        else:
            r, v = strategy_fn()
            all_rewards.append(r)
            all_final_values.append(v)

    # Average across runs at each step
    avg = [sum(all_rewards[run][step] for run in range(runs)) / runs
           for step in range(NUM_STEPS)]

    if all_final_values:
        avg_values = [sum(all_final_values[run][arm] for run in range(runs)) / runs
                      for arm in range(NUM_ARMS)]
        return avg, avg_values
    return avg, None


def print_bar_chart(label, values, true_means):
    """Print a simple ASCII bar chart comparing estimated vs true arm values."""
    print(f"\n{label} — Learned arm estimates vs true means:")
    print(f"  {'Arm':>4} | {'Estimated':>10} | {'True Mean':>10} | Bar")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+----------")
    for i, (est, true) in enumerate(zip(values, true_means)):
        bar = "█" * int(est * 20)
        best = " ← BEST" if true == max(true_means) else ""
        print(f"  {i:>4} | {est:>10.3f} | {true:>10.3f} | {bar}{best}")


def print_performance_summary(eg_avg, ucb_avg, rand_avg):
    """Print final average reward at key checkpoints."""
    checkpoints = [50, 200, 500, 999]
    print("\nAverage reward over time (higher = better):")
    print(f"  {'Step':>6} | {'Epsilon-Greedy':>15} | {'UCB':>10} | {'Random':>10}")
    print(f"  {'-'*6}-+-{'-'*15}-+-{'-'*10}-+-{'-'*10}")
    for step in checkpoints:
        print(f"  {step+1:>6} | {eg_avg[step]:>15.4f} | {ucb_avg[step]:>10.4f} | {rand_avg[step]:>10.4f}")


def main():
    print("=" * 60)
    print("Multi-Armed Bandit — Strategy Comparison")
    print("Author: BOUTEROUATA Issam Salah Eddine")
    print("=" * 60)
    print(f"\nSetup: {NUM_ARMS} arms, {NUM_STEPS} steps per run, averaged over {NUM_RUNS} runs")
    print(f"True arm means (hidden from agent): {TRUE_MEANS}")
    print(f"Best arm: Arm {TRUE_MEANS.index(max(TRUE_MEANS))} (mean = {max(TRUE_MEANS)})")

    print(f"\nRunning Epsilon-Greedy (ε={EPSILON})...")
    eg_avg, eg_values = average_runs(run_epsilon_greedy)

    print("Running UCB...")
    ucb_avg, ucb_values = average_runs(run_ucb)

    print("Running Random baseline...")
    rand_avg, _ = average_runs(run_random)

    print_performance_summary(eg_avg, ucb_avg, rand_avg)
    print_bar_chart("Epsilon-Greedy", eg_values, TRUE_MEANS)
    print_bar_chart("UCB", ucb_values, TRUE_MEANS)

    # Determine winner
    final_eg  = eg_avg[-1]
    final_ucb = ucb_avg[-1]
    winner = "Epsilon-Greedy" if final_eg > final_ucb else "UCB"
    print(f"\n{'=' * 60}")
    print(f"Winner at step {NUM_STEPS}: {winner}")
    print(f"  Epsilon-Greedy final avg reward: {final_eg:.4f}")
    print(f"  UCB final avg reward:            {final_ucb:.4f}")
    print(f"  Random baseline:                 {rand_avg[-1]:.4f}")
    print(f"  Theoretical best (always pick arm {TRUE_MEANS.index(max(TRUE_MEANS))}): {max(TRUE_MEANS):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
