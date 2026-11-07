# MsPacman validation results (results/mspacman/cluster_seed0)

Wall time of this invocation: 281 s

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
