"""Simulate check-in policies on learner routines; select on validation learners, then test once.

    uv run --locked --extra research python -m experiments.checkin_bandit.run select
    uv run --locked --extra research python -m experiments.checkin_bandit.run test
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import checkin_policy
from checkin_policy import ACTIONS

from . import simulator
from .policies import FixedSchedule, LinUCBPolicy, NeverNudge, Oracle, RandomPolicy, ThompsonPolicy
from .simulator import SLOTS, Learner

RESULTS = Path(__file__).with_name("results")
DAYS = 120
VALIDATION, TEST = (2, 40), (3, 100)  # (seed, learners)
GRID = {"linucb": [0.1, 0.3, 1.0], "thompson": [0.1, 0.3, 1.0]}
PERIODS = ((0, 30), (30, 60), (60, 120))
NUDGE_COSTS = (0.05, 0.15, 0.30)  # sensitivity analysis, declared before validation or test; 0.15 is the main setting


def simulate(policy_cls, parameter, seed, learner_index):
    """One learner for DAYS days. The world's randomness depends only on (seed, learner), so every
    policy meets the same learner; the policy has its own random stream."""
    world = np.random.default_rng([seed, learner_index])
    learner = Learner.sample(world)
    policy = policy_cls(parameter, np.random.default_rng([seed, learner_index, 99]))
    log = []
    for _ in range(DAYS):
        learner.start_day()
        for hour in SLOTS:
            x = learner.features(hour)
            expected = learner.expected_rewards(hour)
            if isinstance(policy, Oracle):
                policy.expected = expected
            action = policy.choose(x, hour)
            reward, engaged = learner.act(hour, action)
            policy.update(action, x, reward)
            best = max(expected.values())
            log.append({"day": learner.day, "action": action, "reward": reward, "engaged": engaged,
                        "regret": best - expected[action],
                        "unnecessary": action != "stay_silent" and expected["stay_silent"] >= expected[action],
                        "obstacle": learner.tired or learner.busy})
        learner.end_day()
    return log


def summarize(logs):
    """Metrics per learner, then averaged; 95% bootstrap interval over learners for cumulative reward."""
    per = []
    for log in logs:
        nudges = [row for row in log if row["action"] != "stay_silent"]
        per.append({
            "cumulative_reward": sum(row["reward"] for row in log),
            "average_regret": float(np.mean([row["regret"] for row in log])),
            "unnecessary_nudge_rate": float(np.mean([row["unnecessary"] for row in nudges])) if nudges else 0.0,
            "ignored_nudge_rate": float(np.mean([not row["engaged"] for row in nudges])) if nudges else 0.0,
            "nudges_per_day": len(nudges) / DAYS,
            "nudges_when_tired_or_busy_rate": float(np.mean([row["obstacle"] for row in nudges])) if nudges else 0.0,
            "engagement_rate": float(np.mean([row["engaged"] for row in log])),
            "regret_by_period": [float(np.mean([r["regret"] for r in log if lo <= r["day"] < hi])) for lo, hi in PERIODS],
        })
    rewards = np.array([p["cumulative_reward"] for p in per])
    rng = np.random.default_rng(0)
    boots = [rewards[rng.integers(0, len(rewards), len(rewards))].mean() for _ in range(2000)]
    out = {key: round(float(np.mean([p[key] for p in per])), 4) for key in per[0] if key != "regret_by_period"}
    out["cumulative_reward_ci95"] = [round(float(x), 2) for x in np.percentile(boots, [2.5, 97.5])]
    out["regret_by_period"] = {f"days {lo + 1}-{hi}": round(float(np.mean([p["regret_by_period"][i] for p in per])), 4)
                               for i, (lo, hi) in enumerate(PERIODS)}
    return out, rewards


def run_policy(policy_cls, parameter, seed, learners):
    return [simulate(policy_cls, parameter, seed, i) for i in range(learners)]


def select():
    start, report = time.time(), {}
    for name, cls in (("linucb", LinUCBPolicy), ("thompson", ThompsonPolicy)):
        trials = {}
        for parameter in GRID[name]:
            summary, _ = summarize(run_policy(cls, parameter, *VALIDATION))
            trials[parameter] = summary["cumulative_reward"]
            print(f"  {name} {parameter}: validation mean cumulative reward {trials[parameter]:.2f}")
        report[name] = {"validation_cumulative_reward": trials, "chosen": max(trials, key=trials.get)}
    report["seconds"] = round(time.time() - start, 1)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["chosen"] for k, v in report.items() if k != "seconds"}))


def test():
    selection = json.loads((RESULTS / "selection.json").read_text())
    if selection["linucb"]["chosen"] != checkin_policy.ALPHA:
        raise SystemExit("checkin_policy.ALPHA differs from the validation selection; update it first.")
    if (RESULTS / "test.json").exists():
        raise SystemExit("Test learners were already simulated; refusing to reopen them.")
    policies = {"linucb": (LinUCBPolicy, selection["linucb"]["chosen"]),
                "thompson": (ThompsonPolicy, selection["thompson"]["chosen"]),
                "fixed_schedule": (FixedSchedule, None), "random": (RandomPolicy, None),
                "never_nudge": (NeverNudge, None), "oracle": (Oracle, None)}
    report, rewards = {"protocol": {"days": DAYS, "decisions_per_day": len(SLOTS), "test": dict(zip(("seed", "learners"), TEST))}}, {}
    for name, (cls, parameter) in policies.items():
        report[name], rewards[name] = summarize(run_policy(cls, parameter, *TEST))
        report[name]["parameter"] = parameter
        print(name, {k: report[name][k] for k in ("cumulative_reward", "average_regret", "unnecessary_nudge_rate", "nudges_per_day")})
    # Paired comparison on the same learners: learner-bootstrap interval for the reward difference.
    rng = np.random.default_rng(1)
    for other in ("thompson", "fixed_schedule", "random", "never_nudge"):
        diff = rewards["linucb"] - rewards[other]
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        report.setdefault("paired_reward_difference", {})[f"linucb_minus_{other}"] = {
            "mean": round(float(diff.mean()), 2), "ci95": [round(float(x), 2) for x in np.percentile(boots, [2.5, 97.5])],
            "learners_better": int((diff > 0).sum())}
    # Sensitivity: the same test learners and chosen parameters under other nudge costs.
    report["nudge_cost_sensitivity"] = {}
    main_cost = simulator.NUDGE_COST
    for cost in NUDGE_COSTS:
        simulator.NUDGE_COST = cost
        report["nudge_cost_sensitivity"][str(cost)] = {
            name: {k: summarize(run_policy(cls, parameter, *TEST))[0][k]
                   for k in ("cumulative_reward", "average_regret", "unnecessary_nudge_rate", "nudges_per_day")}
            for name, (cls, parameter) in policies.items()}
    simulator.NUDGE_COST = main_cost
    (RESULTS / "test.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["select", "test"])
    {"select": select, "test": test}[parser.parse_args().phase]()


if __name__ == "__main__":
    main()
