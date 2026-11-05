# MsPacman validation results (results/mspacman/main_seed0)

Wall time of this invocation: 307 s

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

## run

```
"results/mspacman/main_seed0"
```

## P3_snapshots

```
[
 {
  "episode": 0,
  "mean": 332.0,
  "std": 77.56287771866127,
  "se": 17.34358671094304,
  "length": 522.6
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
  "mean": 335.0,
  "std": 340.63910521254013,
  "se": 76.16921950499426,
  "length": 552.2
 },
 {
  "episode": 500,
  "mean": 461.5,
  "std": 212.58586500517856,
  "se": 47.53564452071729,
  "length": 618.6
 },
 {
  "episode": 1000,
  "mean": 1000.0,
  "std": 784.8757863509359,
  "se": 175.50356121742942,
  "length": 687.5
 },
 {
  "episode": 1100,
  "mean": 891.0,
  "std": 331.5855847288902,
  "se": 74.14479078128146,
  "length": 739.4
 }
]
```

## P4_calibration

```
{
 "steps": 15142,
 "mean_pred": 6.821261882781982,
 "mean_true": 6.5492937526987545,
 "mean_bias": 0.2719679573470624,
 "corr": 0.24903562068012788,
 "bins_pred": [
  1.8509999513626099,
  2.180000066757202,
  3.697999954223633,
  4.324999809265137,
  4.913000106811523,
  5.4039998054504395,
  5.802999973297119,
  6.159999847412109,
  6.431000232696533,
  6.671999931335449,
  6.876999855041504,
  7.085000038146973,
  7.328999996185303,
  7.642000198364258,
  8.043999671936035,
  8.543000221252441,
  9.097999572753906,
  9.968000411987305,
  11.163999557495117,
  13.196000099182129
 ],
 "bins_true": [
  13.12,
  11.277,
  3.653,
  9.009,
  3.212,
  1.667,
  2.436,
  1.471,
  1.778,
  2.594,
  3.185,
  3.747,
  4.126,
  4.564,
  5.851,
  8.213,
  10.226,
  11.842,
  12.884,
  16.07
 ]
}
```

## P5_events

```
{
 "life lost": {
  "n": 40,
  "q_at_-60": 7.3606743812561035,
  "q_at_-10": 6.565113067626953,
  "q_at_0": 5.8553996086120605,
  "q_at_+10": 4.042421817779541
 },
 "power pellet eaten": {
  "n": 21,
  "q_at_-60": 7.924477577209473,
  "q_at_-10": 7.813259124755859,
  "q_at_0": 7.177025318145752,
  "q_at_+10": 7.011632919311523
 },
 "ghost eaten": {
  "n": 17,
  "q_at_-60": 7.143918037414551,
  "q_at_-10": 6.62860631942749,
  "q_at_0": 6.633908271789551,
  "q_at_+10": 6.548770427703857
 }
}
```

## P9_composition_final

```
{
 "games": 20,
 "mean_score": 985.0,
 "reward_values": {
  "10": 1392,
  "50": 21,
  "100": 5,
  "200": 10,
  "210": 3,
  "400": 4
 },
 "share_of_score_from_rewards_>=100": 0.24010152284263958,
 "ghosts_per_game": 0.85,
 "power_pellets_per_game": 1.05
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
| 0 | 3 | (103, 80) | 1.970 | UPLEFT | 0.0678 | UPLEFT | 0 | 1.924 |
| 1 | 3 | (103, 79) | 1.944 | UPLEFT | 0.0789 | UPLEFT | 0 | 2.012 |
| 2 | 3 | (103, 80) | 2.032 | UPLEFT | 0.0848 | UPLEFT | 0 | 1.772 |
| 3 | 3 | (103, 80) | 1.790 | UPLEFT | 0.0744 | UPLEFT | 0 | 1.818 |
| 4 | 3 | (103, 80) | 1.836 | UPLEFT | 0.0747 | UPLEFT | 0 | 1.921 |
| 5 | 3 | (103, 79) | 1.940 | UPLEFT | 0.0823 | UPLEFT | 0 | 1.898 |
| 6 | 3 | (103, 80) | 1.917 | UPLEFT | 0.0771 | UPLEFT | 0 | 1.803 |
| 7 | 3 | (103, 80) | 1.821 | UPLEFT | 0.0781 | UPLEFT | 0 | 1.829 |
| 8 | 3 | (103, 80) | 1.847 | UPLEFT | 0.0789 | UPLEFT | 0 | 1.976 |
| 9 | 3 | (103, 79) | 1.996 | UPLEFT | 0.0792 | UPLEFT | 0 | 1.922 |

## P7_determinism

```
{
 "sticky_0.0": {
  "scores": [
   870.0,
   870.0,
   870.0,
   870.0,
   870.0
  ],
  "lengths": [
   779,
   779,
   779,
   779,
   779
  ]
 },
 "sticky_0.25": {
  "scores": [
   600.0,
   710.0,
   1140.0,
   780.0,
   1040.0
  ],
  "lengths": [
   745,
   577,
   633,
   783,
   783
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
 "action_gap_frozen_mean": 0.0809555577509331,
 "action_gap_after_mean": 0.375414804176048,
 "log_maxq_growth_per_frozen_step": 0.00021664558992622564,
 "predicted_growth": 0.01005033585350145,
 "maxq_first_and_last_frozen_step": [
  1.9704378843307495,
  2.1590282917022705
 ]
}
```

## P8_positions

```
{
 "episode 250": {
  "positions": 5912,
  "fraction_of_4x4_cells_visited": 0.07267441860465117
 },
 "final network": {
  "positions": 8430,
  "fraction_of_4x4_cells_visited": 0.2453488372093023
 }
}
```
