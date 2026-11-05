# Preliminary CartPole runs (tuning, made before the final configuration)

Made with earlier versions of `dqn/cartpole_dqn.py` in the same session; only logs and summaries are kept.

| folder | configuration | outcome (500 greedy roll-outs) |
|---|---|---|
| cp_try0 | 600 episodes, epsilon floor 0.02, raw inputs, 128x128 | final network 114.8 mean; all 500 episodes ended with the cart off the track |
| cp_v1 / cp_v2 | 1500 episodes, raw inputs, 128x128 / 256x256 | 444.5 / 448.8 mean; 40.6 % / 59.8 % reach 500 steps |
| tA..tD | 1500 episodes, scaled inputs; tB buffer 200k + batch 128, tC soft target tau 0.005, tD 256x256 | greedy checks oscillate between ~100 and 500 in all four (killed before the roll-outs) |
| es0..es3 | stop after 50 consecutive 500-step greedy evaluation episodes, rule armed from the start | stopped at episodes 125-250 with epsilon 0.75-0.91 and max Q 3.4-8.9; all four 500.0 in all 500 roll-outs |
| eg0..eg3 | same rule, armed only once epsilon reached its floor of 0.02 | stopped at episodes 1025, 900, 800 (97.4 %, 98.2 %, 99.0 % at 500 steps); seed 1 never stopped (final 291.7, best check network 500.0) |
| ef0..ef3 | epsilon floor 0.1, 1500 episodes, rule with 100 episodes (never triggered) | best-check networks 500.0 in all 500 roll-outs for all four seeds; final networks 110-172 |
