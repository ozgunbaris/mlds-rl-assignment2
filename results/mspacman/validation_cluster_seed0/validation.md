# MsPacman validation results (results/mspacman/cluster_seed0)

Wall time of this invocation: 242 s

## P1_preprocessing

```
{
 "frames": 449,
 "pixels_with_sum_448_in_crop": {
  "min": 7,
  "max": 17,
  "mean": 10.703786191536748,
  "frames_with_none": 0
 },
 "pixels_with_sum_448_that_are_not_mspacman_colour_max": 0,
 "value_range_preprocessed": [
  -128,
  27
 ]
}
```

## P2_random

```
{
 "games": 100,
 "mean": 245.1,
 "std": 50.44789391044983,
 "min": 140.0,
 "max": 410.0,
 "length_mean": 494.64
}
```

## P3_snapshots

```
[
 {
  "episode": 0,
  "mean": 210.0,
  "std": 0.0,
  "se": 0.0,
  "length": 375.0
 },
 {
  "episode": 100,
  "mean": 60.0,
  "std": 0.0,
  "se": 0.0,
  "length": 483.0
 },
 {
  "episode": 250,
  "mean": 60.0,
  "std": 0.0,
  "se": 0.0,
  "length": 465.0
 },
 {
  "episode": 500,
  "mean": 593.5,
  "std": 436.4318388935436,
  "se": 97.58912593112001,
  "length": 582.7
 },
 {
  "episode": 1000,
  "mean": 717.5,
  "std": 83.71827757425496,
  "se": 18.719975961523026,
  "length": 699.9
 },
 {
  "episode": 1500,
  "mean": 1052.0,
  "std": 212.92252111977257,
  "se": 47.61092311644461,
  "length": 756.1
 },
 {
  "episode": 2000,
  "mean": 929.0,
  "std": 206.27408950229304,
  "se": 46.1242886124003,
  "length": 648.2
 },
 {
  "episode": 2500,
  "mean": 1337.5,
  "std": 584.5757008292425,
  "se": 130.71510050487663,
  "length": 722.7
 },
 {
  "episode": 3000,
  "mean": 1293.0,
  "std": 450.9556519215609,
  "se": 100.83674925343438,
  "length": 800.5
 },
 {
  "episode": 4000,
  "mean": 1384.5,
  "std": 482.32224705066216,
  "se": 107.8505331465728,
  "length": 804.6
 },
 {
  "episode": 5000,
  "mean": 1928.5,
  "std": 616.386850930485,
  "se": 137.82828991175938,
  "length": 1053.2
 },
 {
  "episode": 6000,
  "mean": 2111.0,
  "std": 371.9126241471241,
  "se": 83.16219092832992,
  "length": 990.1
 }
]
```

## P4_calibration

```
{
 "steps": 19484,
 "mean_pred": 12.456392288208008,
 "mean_true": 12.251511370755301,
 "mean_bias": 0.2048805067836159,
 "corr": 0.7523833494864447,
 "bins_pred": [
  1.184000015258789,
  1.9830000400543213,
  2.4739999771118164,
  3.2739999294281006,
  3.928999900817871,
  4.650000095367432,
  5.703999996185303,
  6.801000118255615,
  7.9019999504089355,
  9.397000312805176,
  11.057000160217285,
  13.027000427246094,
  14.85099983215332,
  16.91699981689453,
  19.30699920654297,
  21.881999969482422,
  24.069000244140625,
  25.749000549316406,
  27.020000457763672,
  27.898000717163086
 ],
 "bins_true": [
  5.663,
  17.275,
  6.568,
  1.469,
  2.211,
  5.975,
  5.266,
  4.7,
  5.049,
  6.542,
  7.792,
  10.601,
  14.033,
  14.201,
  16.356,
  19.723,
  22.899,
  24.901,
  26.255,
  27.255
 ]
}
```

## P5_events

```
{
 "life lost": {
  "n": 40,
  "q_at_-60": 10.301603317260742,
  "q_at_-10": 3.961392641067505,
  "q_at_0": 2.622267246246338,
  "q_at_+10": 4.433039665222168
 },
 "power pellet eaten": {
  "n": 77,
  "q_at_-60": 17.559642791748047,
  "q_at_-10": 23.38312530517578,
  "q_at_0": 23.247831344604492,
  "q_at_+10": 22.410770416259766
 },
 "ghost eaten": {
  "n": 44,
  "q_at_-60": 20.350955963134766,
  "q_at_-10": 17.33624267578125,
  "q_at_0": 16.903032302856445,
  "q_at_+10": 17.329761505126953
 }
}
```

## P9_composition_final

```
{
 "games": 20,
 "mean_score": 2242.0,
 "reward_values": {
  "10": 2760,
  "50": 77,
  "100": 10,
  "110": 2,
  "200": 13,
  "210": 17,
  "400": 13,
  "800": 1
 },
 "share_of_score_from_rewards_>=100": 0.2986173059768064,
 "ghosts_per_game": 2.2,
 "power_pellets_per_game": 3.85
}
```

## P9_composition_random

```
{
 "games": 20,
 "mean_score": 272.5,
 "reward_values": {
  "10": 515,
  "50": 2,
  "200": 1
 },
 "share_of_score_from_rewards_>=100": 0.03669724770642202,
 "ghosts_per_game": 0.05,
 "power_pellets_per_game": 0.1
}
```

## P6 first 10 steps

| t | lives | Ms. Pac-Man (row, col) | max Q | argmax | max Q - min Q | action | reward | y = clip(r) + 0.99 max Q(s') |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | (103, 80) | 1.774 | UPRIGHT | 0.0697 | UPRIGHT | 0 | 1.781 |
| 1 | 3 | (103, 79) | 1.799 | UPRIGHT | 0.0791 | UPRIGHT | 0 | 1.874 |
| 2 | 3 | (103, 80) | 1.893 | UPRIGHT | 0.0802 | UPRIGHT | 0 | 1.816 |
| 3 | 3 | (103, 80) | 1.834 | DOWN | 0.0880 | DOWN | 0 | 1.761 |
| 4 | 3 | (103, 80) | 1.779 | LEFT | 0.0732 | LEFT | 0 | 1.903 |
| 5 | 3 | (103, 79) | 1.923 | DOWN | 0.0654 | DOWN | 0 | 1.953 |
| 6 | 3 | (103, 80) | 1.973 | DOWN | 0.0688 | DOWN | 0 | 1.908 |
| 7 | 3 | (103, 80) | 1.927 | LEFT | 0.0738 | LEFT | 0 | 1.807 |
| 8 | 3 | (103, 80) | 1.825 | LEFT | 0.0589 | LEFT | 0 | 1.870 |
| 9 | 3 | (103, 79) | 1.889 | LEFT | 0.0818 | LEFT | 0 | 1.964 |

## P7_determinism

```
{
 "sticky_0.0": {
  "scores": [
   2490.0,
   2490.0,
   2490.0,
   2490.0,
   2490.0
  ],
  "lengths": [
   1239,
   1239,
   1239,
   1239,
   1239
  ]
 },
 "sticky_0.25": {
  "scores": [
   1480.0,
   1880.0,
   1840.0,
   2070.0,
   2450.0
  ],
  "lengths": [
   701,
   1047,
   1025,
   875,
   863
  ]
 }
}
```

## P7_frozen_start

```
{
 "steps_checked": 120,
 "frozen_steps": 66,
 "first_step_where_actions_matter": 66,
 "action_gap_frozen_mean": 0.07611758239341504,
 "action_gap_after_mean": 0.6816645286701344,
 "log_maxq_growth_per_frozen_step": 0.0008371655567084435,
 "predicted_growth": 0.01005033585350145,
 "maxq_first_and_last_frozen_step": [
  1.7741245031356812,
  2.1323838233947754
 ]
}
```

## P8_positions

```
{
 "episode 250": {
  "positions": 4650,
  "fraction_of_4x4_cells_visited": 0.00813953488372093
 },
 "final network": {
  "positions": 10142,
  "fraction_of_4x4_cells_visited": 0.24593023255813953
 }
}
```

## run

```
"results/mspacman/cluster_seed0"
```

## P10_input_coverage

```
{
 "states": 4655,
 "zero_gradient_rows_cols": {
  "nature": {
   "rows": [
    84,
    85,
    86,
    87
   ],
   "cols": [
    76,
    77,
    78,
    79
   ]
  },
  "nips": {
   "rows": [
    84,
    85,
    86,
    87
   ],
   "cols": [
    76,
    77,
    78,
    79
   ]
  }
 },
 "noise_test": {
  "rows 84-87": {
   "max_abs_dq": 0.0,
   "greedy_action_changed_share": 0.0
  },
  "cols 76-79": {
   "max_abs_dq": 0.0,
   "greedy_action_changed_share": 0.0
  },
  "rows 80-83": {
   "max_abs_dq": 9.740850448608398,
   "greedy_action_changed_share": 0.37636949516648766
  },
  "cols 72-75": {
   "max_abs_dq": 12.269116401672363,
   "greedy_action_changed_share": 0.5544575725026852
  }
 },
 "lives_indicator_pixels_in_preprocessed_rows_86_87": {
  "3": 8,
  "2": 4,
  "1": 0
 },
 "lives_redraw_test": {
  "3 lives drawn as 1": {
   "states": 2727,
   "pixels_changed_per_frame": 8,
   "max_abs_dq": 0.0,
   "greedy_action_changed_share": 0.0
  },
  "1 lives drawn as 3": {
   "states": 934,
   "pixels_changed_per_frame": 8,
   "max_abs_dq": 0.0,
   "greedy_action_changed_share": 0.0
  },
  "2 lives drawn as 3": {
   "states": 994,
   "pixels_changed_per_frame": 4,
   "max_abs_dq": 0.0,
   "greedy_action_changed_share": 0.0
  },
  "3 lives drawn as 2": {
   "states": 2727,
   "pixels_changed_per_frame": 4,
   "max_abs_dq": 0.0,
   "greedy_action_changed_share": 0.0
  }
 },
 "mspacman_share_of_steps_partly_in_raw_cols_152_159": 0.047261009667024706,
 "mspacman_share_of_steps_entirely_in_raw_cols_152_159": 0.002363050483351235,
 "mspacman_lowest_raw_row": 168
}
```

## P11_frozen_phases

```
{
 "games": 5,
 "epsilon": 0.05,
 "start": {
  "n": 5,
  "steps": {
   "mean": 66.0,
   "min": 66.0,
   "max": 66.0
  },
  "distinct_states": {
   "mean": 19.0,
   "min": 19.0,
   "max": 19.0
  },
  "maxq_mean": {
   "mean": 1.9735654610576052,
   "min": 1.9735654610576052,
   "max": 1.9735654610576052
  },
  "maxq_first": {
   "mean": 1.7741245031356812,
   "min": 1.7741245031356812,
   "max": 1.7741245031356812
  },
  "maxq_last": {
   "mean": 2.1323838233947754,
   "min": 2.1323838233947754,
   "max": 2.1323838233947754
  },
  "realised_return_first_step": {
   "mean": 12.622461248433131,
   "min": 9.609949679611608,
   "max": 13.672644213365192
  },
  "maxq_mean_steps_5_to_14_after": {
   "mean": 27.957606887817384,
   "min": 27.957606887817384,
   "max": 27.957606887817384
  },
  "huber_fixed_point": 1.538461538461537,
  "squared_loss_fixed_point": 16.774564132690426
 },
 "restart_after_lost_life": {
  "n": 10,
  "steps": {
   "mean": 29.0,
   "min": 29.0,
   "max": 29.0
  },
  "distinct_states": {
   "mean": 19.0,
   "min": 19.0,
   "max": 19.0
  },
  "maxq_mean": {
   "mean": 4.516106065388383,
   "min": 3.5991332284335433,
   "max": 5.586379560930975
  },
  "realised_return_first_step": {
   "mean": 7.738248899087145,
   "min": 0.0,
   "max": 16.05518541151181
  },
  "maxq_mean_steps_5_to_14_after": {
   "mean": 12.915946547985076,
   "min": 3.89038827419281,
   "max": 24.75148162841797
  },
  "huber_fixed_point": 3.571428571428568,
  "squared_loss_fixed_point": 9.989677408207205
 },
 "death_animation": {
  "n": 15,
  "steps": {
   "mean": 19.533333333333335,
   "min": 19.0,
   "max": 20.0
  },
  "reward_sum": {
   "mean": 0.0,
   "min": 0.0,
   "max": 0.0
  },
  "maxq_mean": {
   "mean": 4.525737416955986,
   "min": 0.6954047460305063,
   "max": 14.273034748278166
  }
 },
 "steps_with_prediction_below_3_and_return_above_10": 331,
 "of_these_in_the_frozen_start": 326
}
```

## P11_corridor_test

```
{
 "n=66, huber": {
  "learned_per_seed": [
   1.5034816950559615,
   1.5768098071217538,
   1.3732082352042199
  ],
  "predicted": 1.538461538461537,
  "true_values_range": [
   13.00851306625766,
   25.0
  ]
 },
 "n=66, mse": {
  "learned_per_seed": [
   14.626963818073273,
   15.21329061985016,
   14.031032049655915
  ],
  "predicted": 15.151515151515147,
  "true_values_range": [
   13.00851306625766,
   25.0
  ]
 },
 "n=29, huber": {
  "learned_per_seed": [
   3.5728097766637803,
   3.2742220878601076,
   3.38811956346035
  ],
  "predicted": 3.571428571428568,
  "true_values_range": [
   18.867982180090813,
   25.0
  ]
 },
 "n=29, mse": {
  "learned_per_seed": [
   19.39518723487854,
   19.0174654006958,
   19.01920986175537
  ],
  "predicted": 19.531249999999996,
  "true_values_range": [
   18.867982180090813,
   25.0
  ]
 }
}
```
