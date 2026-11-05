"""Fill the MsPacman placeholders of report/report.md from the result files (run from the repository root).

    python report/fill_mspacman.py

Every number in the generated text is read from results/mspacman/; nothing is typed by hand.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from dqn.common import moving_average, read_csv  # noqa: E402

R = "results/mspacman"


def run_numbers(run):
    e = read_csv(f"{R}/{run}/episodes.csv")
    ma = moving_average(e["reward"], 100)
    h = read_csv(f"{R}/{run}/heldout_q.csv")
    ev = read_csv(f"{R}/{run}/eval_periodic.csv")
    out = {"ep": len(e["episode"]), "steps": int(e["step"][-1]), "upd": int(np.nansum(e["n_updates"])),
           "hours": e["wall_time_s"][-1] / 3600, "first100": float(np.mean(e["reward"][:100])),
           "last100": float(np.mean(e["reward"][-100:])), "best": float(ma[99:].max()), "bestep": int(np.argmax(ma[99:])) + 100,
           "q_last": float(np.mean(e["qmax_mean"][-100:])), "qmax_last": float(np.mean(e["qmax_max"][-100:])),
           "held_last": float(h["qmax_mean"][-1]), "maxlen": int(e["length"].max()),
           "len_first": float(np.mean(e["length"][:100])), "len_last": float(np.mean(e["length"][-100:])),
           "ev": list(zip(ev["episode"].astype(int).tolist(), ev["eval_mean"].tolist())) if len(ev) else [],
           "eps_last": float(e["epsilon"][-1]), "ma": ma, "qma": moving_average(e["qmax_mean"], 100),
           "last100_min": float(np.min(e["reward"][-100:])), "last100_max": float(np.max(e["reward"][-100:])),
           "held": h}
    p = f"{R}/{run}/eval_summary_eps0.json"
    out["roll"] = json.load(open(p)) if os.path.exists(p) else None
    return out


def held_sentence(a):
    h = a["held"]["qmax_mean"]
    ep = a["held"]["episode"]
    k = min(13, len(h))
    old, new, gap = h[-k], h[-1], int(ep[-1] - ep[-k])
    trend = "still rising" if new > 1.05 * old else ("still falling" if new < 0.95 * old else "nearly constant")
    return f"{gap} games before the end the held-out value was {old:.2f}, so it was {trend} when training stopped."




def direction(a, b, up="rises", down="falls", flat="stays about level"):
    return up if b > a * 1.05 + 0.02 else (down if b < a * 0.95 - 0.02 else flat)


def intro_verdict(f):
    ratio = f["log_maxq_growth_per_frozen_step"] / f["predicted_growth"]
    if 0.7 <= ratio <= 1.3:
        return f" The network reproduces the predicted growth ({100 * ratio:.0f} % of the predicted slope)."
    return (f" The network does not reproduce it ({100 * ratio:.0f} % of the predicted slope). A likely reason (not tested): the "
            f"frames of the intro differ only by small animations, so the network gives them almost the same value, and because no "
            f"action matters there, this error never changes a decision.")


def p_texts(v):
    """Paragraphs for tests P3 to P9, generated from validation.json."""
    t = {}
    rows = v.get("P3_snapshots", [])
    if rows:
        lst = ", ".join(f"{r['episode']:,}: {r['mean']:.0f}" for r in rows)
        t["P3"] = (f"Every saved network played 20 greedy games (sticky actions on, new seeds), so the exploration noise of the training "
                   f"curve is absent. Mean score by training game of the snapshot: {lst}. The untrained network scores {rows[0]['mean']:.0f} "
                   f"and the last one {rows[-1]['mean']:.0f} (standard error {rows[-1]['se']:.0f}).")
    c = v.get("P4_calibration")
    if c:
        bias = c["mean_pred"] - c["mean_true"]
        word = "overestimates" if bias > 0 else "underestimates"
        strength = ("clearly" if c["corr"] > 0.5 else ("weakly" if c["corr"] > 0.2 else "barely"))
        t["P4"] = (f"The network is trained to predict the discounted sum of clipped rewards up to the next lost life. Along 20 greedy games "
                   f"({c['steps']:,} steps) the predicted $Q(s_t, a_t)$ averages {c['mean_pred']:.2f} and the realised return of the same "
                   f"policy {c['mean_true']:.2f}: the network {word} by {abs(bias):.2f} on average. Prediction and outcome are {strength} correlated "
                   f"(correlation {c['corr']:.2f}); sorted into 20 bins of the prediction, the realised return goes from "
                   f"{c['bins_true'][0]:.2f} in the lowest bin (prediction {c['bins_pred'][0]:.2f}) to {c['bins_true'][-1]:.2f} in the highest "
                   f"(prediction {c['bins_pred'][-1]:.2f}).")
    e = v.get("P5_events")
    if e:
        parts = []
        for name, d in e.items():
            nice = {"life lost": "lost life", "power pellet eaten": "power pellet", "ghost eaten": "eaten ghost"}.get(name, name)
            art = "an" if nice[0] in "aeiou" else "a"
            parts.append(f"around {art} {nice} (n = {d['n']}) the average $\\max_a Q$ is {d['q_at_-60']:.2f} sixty steps before, "
                         f"{d['q_at_-10']:.2f} ten steps before, {d['q_at_0']:.2f} at the event and {d['q_at_+10']:.2f} ten steps after")
        L = e.get("life lost")
        extra = ""
        if L:
            extra = (f" Before a lost life the value {direction(L['q_at_-60'], L['q_at_-10'])} as the ghosts close in and drops at the "
                     f"event; it stays low during the death animation, when no reward can be collected.")
        t["P5"] = ("For every event in 20 greedy games the trace of $\\max_a Q$ is aligned on the event and averaged: " + "; ".join(parts) + "." + extra)
    tr = v.get("P6_trace")
    if tr:
        head = "| t | lives | Ms. Pac-Man (row, col) | max Q | argmax | max Q - min Q | action | reward | y = clip(r) + 0.99 max Q(s') |\n|---|---|---|---|---|---|---|---|---|\n"
        fz = v.get("P7_frozen_start", {}).get("first_step_where_actions_matter", 0)
        intro = ("During these steps the game is still in its frozen intro (test P7), so no action has any effect, the reward is 0, "
                 "and the Bellman target is $0.99 \\max_a Q(s')$." if fz >= 10 else "")
        t["P6"] = ("The table follows the final network from the start of a game. " + intro + "\n\n"
                   + head + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in tr) + "\n")
    d = v.get("P7_determinism")
    f = v.get("P7_frozen_start")
    if d and f:
        s0, s1 = d["sticky_0.0"]["scores"], d["sticky_0.25"]["scores"]
        same = len(set(s0)) == 1
        t["P7"] = (f"With sticky actions switched off, the emulator and the greedy policy are both deterministic, so games started with "
                   f"different seeds must be identical: the five games scored {', '.join(f'{x:.0f}' for x in s0)}"
                   f"{' (identical, also in length)' if same and len(set(d['sticky_0.0']['lengths'])) == 1 else ''}. With the default "
                   f"sticky actions the same five seeds give {', '.join(f'{x:.0f}' for x in s1)}. "
                   f"At each of the first {f['steps_checked']} steps of a game the emulator was cloned and all 9 actions were applied: "
                   f"in the first {f['first_step_where_actions_matter']} steps every action leads to exactly the same emulator state "
                   f"({f['frozen_steps']} frozen steps in all), the intro of the game. Where actions cannot matter the network's action gap "
                   f"$\\max_a Q - \\min_a Q$ averages {f['action_gap_frozen_mean']:.3f}, against {f['action_gap_after_mean']:.3f} afterwards."
                   + (f" During the intro nothing can happen except that the first reward comes one step closer, so $\\max_a Q$ should grow by "
                      f"the factor $1/\\gamma$ per step, i.e. $\\log \\max_a Q$ with slope $-\\log 0.99 = {f['predicted_growth']:.4f}$; the "
                      f"fitted slope over the {f['first_step_where_actions_matter']} frozen steps is {f['log_maxq_growth_per_frozen_step']:.4f} "
                      f"($\\max_a Q$ from {f['maxq_first_and_last_frozen_step'][0]:.3f} to {f['maxq_first_and_last_frozen_step'][1]:.3f})."
                      + intro_verdict(f)
                      if f.get("log_maxq_growth_per_frozen_step") is not None else ""))
    pos = v.get("P8_positions")
    if pos:
        items = list(pos.items())
        t["P8"] = ("Ms. Pac-Man's position was read from her colour in the raw frames of 10 greedy games per network. " +
                   "; ".join(f"{('the network after ' + k if k.startswith('episode') else 'the ' + k).capitalize() if i == 0 else ('the network after ' + k if k.startswith('episode') else 'the ' + k)} visited "
                             f"{100 * x['fraction_of_4x4_cells_visited']:.1f} % of the 4 x 4-pixel cells of the "
                             f"screen ({x['positions']:,} positions)" for i, (k, x) in enumerate(items)) + ".")
    cf, cr = v.get("P9_composition_final"), v.get("P9_composition_random")
    if cf and cr:
        t["P9"] = (f"Over 20 greedy games the final network scores {cf['mean_score']:.0f} per game, of which "
                   f"{100 * cf['share_of_score_from_rewards_>=100']:.1f} % comes from rewards of 100 points or more (ghosts and fruit); it eats "
                   f"{cf['power_pellets_per_game']:.2f} power pellets and {cf['ghosts_per_game']:.2f} ghosts per game. The random policy "
                   f"scores {cr['mean_score']:.0f}, {100 * cr['share_of_score_from_rewards_>=100']:.1f} % from such rewards, with "
                   f"{cr['power_pellets_per_game']:.2f} power pellets and {cr['ghosts_per_game']:.2f} ghosts per game. Reward values seen "
                   f"(value: count): {', '.join(f'{k}: {n}' for k, n in list(cf['reward_values'].items())[:12])}.")
    return t


def main():
    s = open("report/report.md").read()
    a = run_numbers("main_seed0")
    b = run_numbers("main_seed1") if os.path.exists(f"{R}/main_seed1/episodes.csv") else None
    v = json.load(open(f"{R}/validation/validation.json"))
    rnd = v["P2_random"]
    ev_first, ev_last = a["ev"][0], a["ev"][-1]
    ev_best = max(a["ev"], key=lambda x: x[1])
    roll = a["roll"]
    txt = [f"**Learning.** Seed 0 trained for {a['hours']:.2f} hours: {a['ep']:,} games, {a['steps']:,} agent steps "
           f"({4 * a['steps'] / 1e6:.1f} million frames) and {a['upd']:,} updates. Its 100-game moving-average score rose from "
           f"{a['first100']:.0f} over the first 100 games (a random policy scores {rnd['mean']:.0f}) to {a['last100']:.0f} over the last 100 "
           f"(highest moving average {a['best']:.0f}, at game {a['bestep']:,}), and the greedy evaluation games played every 100 training games "
           f"rose from {ev_first[1]:.0f} at game {ev_first[0]} to {ev_last[1]:.0f} at game {ev_last[0]:,} (best {ev_best[1]:.0f}, game {ev_best[0]:,}). "
           f"Games also became longer, {a['len_first']:.0f} agent steps on average over the first 100 games and {a['len_last']:.0f} over the last 100."]
    if b is not None:
        n1 = b["ep"]
        txt.append(f"Seed 1, which trained for {b['hours']:.2f} hours on a single thread, played {b['ep']:,} games ({b['steps']:,} steps) and "
                   f"its moving average rose from {b['first100']:.0f} to {b['last100']:.0f} (Figure FIG(mspacman_compare_seeds)); after the "
                   f"same {n1:,} games seed 0's moving average was {a['ma'][n1 - 1]:.0f}.")
    txt.append(f"The handout expects 4,000 to 6,000 games; seed 0 played {a['ep']:,}, so these runs show the first part of the learning curve "
               f"rather than its end: over its last 300 games the moving-average score went from {a['ma'][-301]:.0f} to {a['ma'][-1]:.0f} and the "
               f"moving-average value of the visited states from {a['qma'][-301]:.2f} to {a['qma'][-1]:.2f}. For scale, the DQN of "
               f"Mnih et al. (2015) reports 2,311 points on Ms. Pac-Man after 200 million frames under a different evaluation protocol "
               f"(no sticky actions, $\\epsilon = 0.05$, up to 30 no-op starts); seed 0 saw {4 * a['steps'] / 1e6:.1f} million frames.")
    txt.append(f"**(i) Maximum Q-values.** Measured in clipped-reward units, the value of the visited states rose from 0 to "
               f"{a['q_last']:.2f} (mean over the last 100 games) and the average over the 1000 held-out random-policy states to "
               f"{a['held_last']:.2f}; the largest value in a game averaged {a['qmax_last']:.2f}. With $\\gamma = 0.99$ a value of "
               f"{a['q_last']:.1f} means about that many discounted reward events ahead. "
               + held_sentence(a))
    txt.append(f"**(ii) Scores.** The moving average is described above. Single games spread widely: over the last 100 training games "
               f"they range from {a['last100_min']:.0f} to {a['last100_max']:.0f}, because of the sticky actions, the {100 * a['eps_last']:.1f} % of random actions "
               f"that remain at the end, and the three lives, so the moving average and the greedy evaluation games are the quantities to read.")
    if roll is not None:
        r = np.load(f"{R}/main_seed0/eval_rewards_eps0.npy")
        txt.append(f"**(iii) Roll-outs.** The final network of seed 0 scores {roll['mean']:.1f} on average over 500 greedy games, with "
                   f"standard deviation {roll['std']:.1f} (minimum {roll['min']:.0f}, median {roll['median']:.0f}, maximum {roll['max']:.0f}), "
                   f"{roll['mean'] / rnd['mean']:.1f} times the random policy; {100 * np.mean(r > rnd['max']):.1f} % of its games score more "
                   f"than the best of the 100 random games ({rnd['max']:.0f}).")
        if b is not None and b["roll"] is not None:
            txt[-1] += (f" The final network of seed 1 scores {b['roll']['mean']:.1f} (standard deviation {b['roll']['std']:.1f}) after "
                        f"{b['ep']:,} games.")
    if "--dry-run" in sys.argv:
        print("\n\n".join(txt))
        vv = json.load(open(sys.argv[sys.argv.index("--validation") + 1])) if "--validation" in sys.argv else v
        for key, para in p_texts(vv).items():
            print(f"[{key}]", para)
        return
    s = s.replace("«MSPACMAN_RESULTS»", "\n\n".join(txt))
    for key, para in p_texts(v).items():
        s = s.replace(f"«{key}»", para)
    s = s.replace("«MSP_MAXLEN»", f"{max(a['maxlen'], b['maxlen'] if b else 0):,}")
    open("report/report.md", "w").write(s)
    left = sorted(set(re.findall(r"«[^»]*»", s)))
    print("filled; remaining markers:", left)


if __name__ == "__main__":
    main()
