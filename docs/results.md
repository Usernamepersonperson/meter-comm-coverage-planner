# Results (measured by python -m meter_coverage demo)

Town: Murten (FR); 2079 buildings with 4466 assumed meters; 314 candidate sites (194 rooftops, 120 poles). Model: Okumura-Hata suburban + Deygout 3-edge DEM diffraction; fade margin 8.0 dB; entry loss 15.0 dB indoor / 30.0 dB cellar (50% of buildings, assumed). All coverage is modelled, nothing is field-measured.

## Gateways needed (greedy, meter-weighted)

| technology | 80% | 90% | 95% | 99% | 95% with 2-fold | coverage at 60 |
|---|---|---|---|---|---|---|
| lorawan | 3 | 5 | 6 | 12 | 12 | 100.0% |
| wmbus_t1 | 40 | >60 | >60 | >60 | >60 | 88.5% |
| wmbus_c1 | 56 | >60 | >60 | >60 | >60 | 82.1% |
| mioty | 2 | 5 | 6 | 9 | 11 | 100.0% |

## Plans compared (LoRaWAN, one uplink per meter every 15 min)

| plan | gateways | covered | 2-fold | median battery (y) | p10 battery (y) | mean airtime (s) | duty headroom | mean PDR (ALOHA) |
|---|---|---|---|---|---|---|---|---|
| optimised_6 | 6 | 97.7% | 75.5% | 21.2 | 8.0 | 0.142 | 83.5% | 97.22% |
| optimised_12_redundant | 12 | 99.4% | 97.2% | 21.2 | 12.6 | 0.115 | 83.5% | 98.10% |
| tallest_roofs_6 | 6 | 88.8% | 79.8% | 21.2 | 8.0 | 0.163 | 83.5% | 97.55% |
| demand_centroids_6 | 6 | 87.5% | 54.2% | 21.2 | 8.0 | 0.181 | 83.5% | 97.58% |

Spreading-factor mix (share of meters):

| plan | SF7 | SF8 | SF9 | SF10 | SF11 | SF12 | uncovered |
|---|---|---|---|---|---|---|---|
| optimised_6 | 74.9% | 7.0% | 5.9% | 5.1% | 3.0% | 1.8% | 2.3% |
| optimised_12_redundant | 79.7% | 5.5% | 6.2% | 5.2% | 2.4% | 0.4% | 0.6% |
| tallest_roofs_6 | 65.7% | 5.3% | 5.6% | 6.3% | 4.0% | 1.9% | 11.2% |
| demand_centroids_6 | 60.1% | 8.5% | 6.7% | 5.9% | 3.0% | 3.3% | 12.5% |

Random 6-site sets (300 draws): covered 70.1% on average (p10 52.6%, p90 83.8%), 2-fold 38.7%.

## Optimiser check (objective = meters covered once + 0.5 x covered twice)

| sites | greedy | greedy+swap | ILP (60 s limit) | ILP upper bound | ILP s |
|---|---|---|---|---|---|
| 3 | 5040.5 | 5265.0 | 5228.5 | 5406.5 | 61.0 |
| 6 | 6022.0 | 6050.5 | 6006.0 | 6202.5 | 60.5 |
| 12 | 6585.0 | 6608.5 | 6606.5 | 6617.0 | 60.8 |

## Model against model (no field data)

| scenario | gateways for 95% | gateways for 95% 2-fold | covered by 5 |
|---|---|---|---|
| Hata suburban + Deygout (base) | 6 | 12 | 93.9% |
| Hata suburban + Epstein-Peterson | 7 | 11 | 94.7% |
| Hata suburban, no terrain | 4 | 7 | 98.1% |
| Hata urban + Deygout | 12 | 24 | 79.5% |
| Hata open + Deygout | 3 | 4 | 99.9% |
| COST-231 (extrapolated) + Deygout | 16 | 31 | 74.9% |
| Free space + Deygout | 1 | 2 | 100.0% |
| Free space only | 1 | 2 | n/a |

Terrain: 72.7% of gateway-building links have a DEM obstruction; mean loss where obstructed Deygout 22.66 dB vs Epstein-Peterson 20.12 dB (mean absolute difference 6.0 dB).

| model | mean offset vs Hata suburban (dB) | sd (dB) |
|---|---|---|
| free space | -30.89 | 5.39 |
| Hata open | -18.38 | 1.0 |
| Hata suburban | 0.0 | 0.0 |
| Hata urban | 9.85 | 0.0 |
| COST-231 (extrapolated) | 12.34 | 0.0 |

## Sensitivity to entry loss

| cellar dB | cellar share | indoor dB | gateways for 95% | 95% 2-fold | covered by 5 | covered by 10 |
|---|---|---|---|---|---|---|
| 20 | 0.5 | 15 | 3 | 6 | 97.8% | 100.0% |
| 30 | 0.5 | 15 | 6 | 12 | 93.9% | 98.8% |
| 40 | 0.5 | 15 | 11 | 24 | 81.2% | 93.1% |
| 30 | 0.2 | 15 | 5 | 9 | 96.2% | 99.6% |
| 30 | 0.8 | 15 | 7 | 12 | 91.7% | 98.2% |
| 30 | 0.5 | 5 | 7 | 11 | 94.6% | 99.1% |
| 30 | 0.5 | 25 | 7 | 12 | 90.1% | 98.4% |

## Period sweep (target plan)

| period (s) | mean PDR | worst gateway PDR | duty headroom | median battery (y) | p10 battery (y) |
|---|---|---|---|---|---|
| 300 | 91.95% | 87.21% | 50.6% | 13.45 | 3.24 |
| 900 | 97.22% | 95.54% | 83.5% | 21.17 | 7.99 |
| 3600 | 99.30% | 98.87% | 95.9% | 26.97 | 17.68 |

## Runtime

Terrain diffraction for all 314 x 2079 links: 105.6 s; greedy curve (60 steps): 0.02 s; swap search: 0.01 s; whole demo: 310.0 s.
