# CartPole validation results

## C1 chain MDP

| variant | max |Q - Q*| per seed | greedy policy optimal |
|---|---|---|
| double_dqn | 0.0017, 0.0000, 0.0000 | True, True, True |
| dqn | 0.0044, 0.0000, 0.0000 | True, True, True |
| no_target | 0.0217, 0.0046, 0.0093 | True, True, True |

| survive chain (T = 50) | mean Q per seed | predicted |
|---|---|---|
| bootstrap_on_truncation, Huber | 20.016, 20.006, 19.992 | 20.000 |
| truncation_as_terminal, Huber | 19.432, 19.870, 19.660 | 19.592 |
| bootstrap_on_truncation, squared | 20.027, 19.964, 19.971 | 20.000 |
| truncation_as_terminal, squared | 15.284, 15.845, 15.219 | 14.493 |
| truncation_as_terminal, squared, no gradient clipping | 14.963, 15.216, 14.810 | 14.493 |

## C2 / C7 run groups

| run | gamma | greedy mean | std | solved | ends | selected net mean | final MA100 | visited max Q | held-out max Q | episodes to MA100 >= 475 |
|---|---|---|---|---|---|---|---|---|---|---|
| main_seed0 | 0.95 | 173.4 | 89.9 | 0.004 | {'angle': 0, 'position': 498, 'time limit': 2} | 500.0 | 125.3 | 19.72 | 18.19 | None |
| main_seed1 | 0.95 | 98.0 | 3.4 | 0.000 | {'angle': 0, 'position': 500, 'time limit': 0} | 495.354 | 85.7 | 19.85 | 18.56 | None |
| main_seed2 | 0.95 | 88.6 | 3.0 | 0.000 | {'angle': 0, 'position': 500, 'time limit': 0} | 455.84 | 89.5 | 19.88 | 18.39 | None |
| main_seed3 | 0.95 | 103.9 | 9.3 | 0.000 | {'angle': 0, 'position': 500, 'time limit': 0} | 500.0 | 88.3 | 19.78 | 18.71 | None |
| main_seed4 | 0.95 | 98.3 | 3.0 | 0.000 | {'angle': 0, 'position': 500, 'time limit': 0} | 500.0 | 101.4 | 20.37 | 18.55 | None |

## C3 overestimation

```
{
 "episodes": 20,
 "lengths_to_2000": [
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000,
  2000
 ],
 "ends": [
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit",
  "time limit"
 ],
 "mean_pred": 20.67240333557129,
 "mean_true": 19.999999999999947,
 "mean_bias": 0.672401807785084,
 "max_abs_bias": 1.0547904968262216,
 "bound": 19.999999999999982
}
```

## C4 first 10 steps

| t | x | x_dot | theta | theta_dot | Q(left) | Q(right) | action | r | y = r + g max Q(s') | y - Q(s,a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.013 | +0.040 | +0.028 | -0.027 | 20.434 | 20.406 | left | 1 | 20.432 | -0.0022 |
| 1 | +0.013 | -0.156 | +0.027 | +0.274 | 20.455 | 20.453 | left | 1 | 20.441 | -0.0135 |
| 2 | +0.010 | -0.351 | +0.032 | +0.575 | 20.465 | 20.420 | left | 1 | 20.427 | -0.0380 |
| 3 | +0.003 | -0.547 | +0.044 | +0.878 | 20.449 | 20.410 | left | 1 | 20.423 | -0.0265 |
| 4 | -0.008 | -0.743 | +0.062 | +1.184 | 19.906 | 20.445 | right | 1 | 20.454 | +0.0087 |
| 5 | -0.023 | -0.548 | +0.085 | +0.911 | 20.477 | 20.412 | left | 1 | 20.698 | +0.2210 |
| 6 | -0.034 | -0.744 | +0.103 | +1.229 | 6.321 | 20.735 | right | 1 | 20.631 | -0.1039 |
| 7 | -0.048 | -0.551 | +0.128 | +0.971 | 8.700 | 20.665 | right | 1 | 20.568 | -0.0968 |
| 8 | -0.059 | -0.358 | +0.147 | +0.721 | 12.637 | 20.598 | right | 1 | 20.578 | -0.0194 |
| 9 | -0.067 | -0.165 | +0.162 | +0.478 | 17.644 | 20.609 | right | 1 | 20.563 | -0.0459 |

## C5 mirror

```
{
 "states": 5000,
 "mean_abs_diff": 0.29425203800201416,
 "max_abs_diff": 2.5405359268188477,
 "corr_action_gap": 0.9702196196003114,
 "greedy_action_agreement": 0.66
}
```
