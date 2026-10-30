# Chapter 4 — Experiments, Results and Analysis (automated report)

*Generated 2026-10-07 07:37:29 by `evaluator.py`.  BRISQUE threshold 45.0; cross-modal IoU threshold 0.95; device cuda.*

## 4.1 Dataset Construction

| Item | Value |
|---|---|
| Background-pool size | 37,852 photographs (nine scene categories) |
| Proof-of-concept batch | 300 RGB-IR pairs (single drone bbox) |
| Annotation type | Single-object bounding box |
| Condition distribution | Horizontal position / vertical position / target scale / time of day |
| End-to-end throughput | ≈90 s/pair (day), ≈150 s/pair (low light) |

## 4.2 Evaluation Metrics

The evaluation is organised around three anchors — alignment precision, generation quality and diversity.  Alignment precision is reported through the bounding-box position error (Eq. 14), the scale error (Eq. 15) and the Intersection-over-Union (Eq. 13), together with a cross-modal check that the RGB and IR boxes coincide.  Generation quality uses the BRISQUE no-reference score (lower is better).  Diversity is quantified through the Shannon entropy (Eq. 16) of the sampled conditions and the number of distinct condition tuples.

## 4.3 Generation-Side Results

| Anchor | Metric | Value |
|---|---|---|
| Alignment | Position error | 1.226 ± 0.448 (min 0.228, max 5.708) px |
| Alignment | Scale error | 0.028 ± 0.021 (min 0.010, max 0.208) |
| Alignment | IoU | 0.706 ± 0.126 (min 0.377, max 0.959) |
| Alignment | Cross-modal IoU | 100% pass (300/300) |
| Quality | BRISQUE | 20.5 ± 20.7 (min -11.2, max 102.0) |
| Quality | Pass rate | 88.3% (265/300) |
| Diversity | Horizontal-position entropy | 3.30 bits |
| Diversity | Vertical-position entropy | 3.30 bits |
| Diversity | Scale entropy | 2.50 bits |
| Diversity | Distinct condition tuples | 300/300 |

### Alignment precision by time of day

| Time of day | n | Position error (px) | Scale error | IoU | BRISQUE | Pass |
|---|---|---|---|---|---|---|
| afternoon | 60 | 1.28 ± 0.39 (min 0.45, max 3.04) | 0.029 ± 0.020 (min 0.010, max 0.125) | 0.692 ± 0.131 (min 0.377, max 0.925) | 16.2 ± 15.9 (min -11.2, max 58.8) | 56/60 |
| dusk | 60 | 1.20 ± 0.39 (min 0.23, max 2.11) | 0.029 ± 0.021 (min 0.010, max 0.150) | 0.697 ± 0.118 (min 0.396, max 0.916) | 20.3 ± 20.2 (min -3.9, max 91.1) | 53/60 |
| night | 60 | 1.19 ± 0.34 (min 0.40, max 1.90) | 0.026 ± 0.015 (min 0.010, max 0.083) | 0.714 ± 0.118 (min 0.417, max 0.954) | 21.2 ± 20.5 (min -6.2, max 90.0) | 53/60 |

## Per-Sample Detail

| ID | Scene | Time of day | Position error (px) | Scale error | IoU | BRISQUE | Pass |
|---|---|---|---|---|---|---|---|
| d0000 | pure_sky | dawn | 1.42 | 0.010 | 0.508 | 12.4 | yes |
| d0001 | coastal | morning | 1.80 | 0.021 | 0.657 | 27.2 | yes |
| d0002 | urban | afternoon | 1.40 | 0.042 | 0.672 | 1.0 | yes |
| d0003 | rural | dusk | 1.02 | 0.011 | 0.908 | 26.3 | yes |
| d0004 | forest | night | 1.05 | 0.027 | 0.875 | 20.7 | yes |
| d0005 | mountain | dawn | 0.74 | 0.050 | 0.468 | 15.9 | yes |
| d0006 | desert | morning | 0.78 | 0.015 | 0.692 | 63.3 | no |
| d0007 | industrial | afternoon | 1.39 | 0.067 | 0.778 | 13.8 | yes |
| d0008 | airfield | dusk | 1.53 | 0.010 | 0.701 | 12.7 | yes |
| d0009 | pure_sky | night | 1.18 | 0.021 | 0.740 | 4.8 | yes |
| d0010 | coastal | dawn | 1.62 | 0.042 | 0.726 | 40.7 | yes |
| d0011 | urban | morning | 0.91 | 0.011 | 0.855 | 70.6 | no |
| d0012 | rural | afternoon | 1.20 | 0.027 | 0.815 | 19.6 | yes |
| d0013 | forest | dusk | 1.70 | 0.050 | 0.599 | 2.1 | yes |
| d0014 | mountain | night | 0.86 | 0.015 | 0.939 | 41.8 | yes |
| d0015 | desert | dawn | 1.44 | 0.033 | 0.826 | 8.6 | yes |
| d0016 | industrial | morning | 1.57 | 0.010 | 0.800 | 22.5 | yes |
| d0017 | airfield | afternoon | 1.41 | 0.021 | 0.554 | 2.5 | yes |
| d0018 | pure_sky | dusk | 1.23 | 0.042 | 0.832 | 3.9 | yes |
| d0019 | coastal | night | 1.47 | 0.011 | 0.925 | 3.9 | yes |
| d0020 | urban | dawn | 0.93 | 0.027 | 0.874 | 14.6 | yes |
| d0021 | rural | morning | 1.31 | 0.050 | 0.754 | 13.7 | yes |
| d0022 | forest | afternoon | 0.79 | 0.015 | 0.651 | 0.1 | yes |
| d0023 | mountain | dusk | 1.52 | 0.033 | 0.604 | -3.8 | yes |
| d0024 | desert | night | 1.39 | 0.010 | 0.873 | 0.2 | yes |
| d0025 | industrial | dawn | 1.38 | 0.021 | 0.793 | 32.0 | yes |
| d0026 | airfield | morning | 1.34 | 0.042 | 0.767 | 16.6 | yes |
| d0027 | pure_sky | afternoon | 1.81 | 0.011 | 0.898 | 34.9 | yes |
| d0028 | coastal | dusk | 0.79 | 0.027 | 0.525 | 67.8 | no |
| d0029 | urban | night | 0.82 | 0.050 | 0.463 | -3.5 | yes |
| d0030 | rural | dawn | 0.58 | 0.015 | 0.651 | 17.7 | yes |
| d0031 | forest | morning | 1.36 | 0.033 | 0.439 | 20.3 | yes |
| d0032 | mountain | afternoon | 1.50 | 0.010 | 0.529 | -3.9 | yes |
| d0033 | desert | dusk | 0.97 | 0.021 | 0.667 | -3.7 | yes |
| d0034 | industrial | night | 1.43 | 0.042 | 0.598 | 20.8 | yes |
| d0035 | airfield | dawn | 1.60 | 0.011 | 0.927 | 28.5 | yes |
| d0036 | pure_sky | morning | 1.15 | 0.027 | 0.780 | 25.4 | yes |
| d0037 | coastal | afternoon | 1.42 | 0.050 | 0.492 | -9.1 | yes |
| d0038 | urban | dusk | 0.23 | 0.015 | 0.879 | 29.9 | yes |
| d0039 | rural | night | 1.77 | 0.033 | 0.646 | 7.9 | yes |
| d0040 | forest | dawn | 1.54 | 0.010 | 0.900 | 43.4 | yes |
| d0041 | mountain | morning | 1.41 | 0.021 | 0.679 | 13.5 | yes |
| d0042 | desert | afternoon | 1.32 | 0.042 | 0.723 | 7.0 | yes |
| d0043 | industrial | dusk | 1.22 | 0.011 | 0.764 | 12.1 | yes |
| d0044 | airfield | night | 0.44 | 0.027 | 0.631 | 23.8 | yes |
| d0045 | pure_sky | dawn | 0.85 | 0.100 | 0.765 | 84.4 | no |
| d0046 | coastal | morning | 1.46 | 0.015 | 0.803 | 6.9 | yes |
| d0047 | urban | afternoon | 1.18 | 0.033 | 0.731 | 4.9 | yes |
| d0048 | rural | dusk | 0.81 | 0.010 | 0.638 | 24.1 | yes |
| d0049 | forest | night | 1.39 | 0.021 | 0.712 | 9.5 | yes |
| d0050 | mountain | dawn | 1.92 | 0.125 | 0.637 | -0.5 | yes |
| d0051 | desert | morning | 1.62 | 0.011 | 0.576 | 10.8 | yes |
| d0052 | industrial | afternoon | 1.01 | 0.027 | 0.622 | 20.6 | yes |
| d0053 | airfield | dusk | 1.30 | 0.050 | 0.460 | 13.5 | yes |
| d0054 | pure_sky | night | 0.85 | 0.015 | 0.606 | 50.8 | no |
| d0055 | coastal | dawn | 1.22 | 0.033 | 0.824 | 17.9 | yes |
| d0056 | urban | morning | 1.45 | 0.010 | 0.628 | 19.6 | yes |
| d0057 | rural | afternoon | 1.61 | 0.021 | 0.655 | 13.9 | yes |
| d0058 | forest | dusk | 1.09 | 0.042 | 0.396 | 18.7 | yes |
| d0059 | mountain | night | 1.11 | 0.011 | 0.929 | 26.9 | yes |
| d0060 | desert | dawn | 0.46 | 0.027 | 0.841 | 51.9 | no |
| d0061 | industrial | morning | 1.44 | 0.050 | 0.777 | 17.3 | yes |
| d0062 | airfield | afternoon | 0.82 | 0.015 | 0.833 | 53.2 | no |
| d0063 | pure_sky | dusk | 1.26 | 0.067 | 0.840 | 55.1 | no |
| d0064 | coastal | night | 0.97 | 0.010 | 0.533 | 6.5 | yes |
| d0065 | urban | dawn | 0.76 | 0.042 | 0.859 | 18.5 | yes |
| d0066 | rural | morning | 1.60 | 0.042 | 0.740 | 13.5 | yes |
| d0067 | forest | afternoon | 1.48 | 0.011 | 0.811 | 3.0 | yes |
| d0068 | mountain | dusk | 1.44 | 0.027 | 0.710 | 39.5 | yes |
| d0069 | desert | night | 1.03 | 0.050 | 0.728 | -0.5 | yes |
| d0070 | industrial | dawn | 0.70 | 0.015 | 0.753 | 2.2 | yes |
| d0071 | airfield | morning | 0.92 | 0.033 | 0.631 | 102.0 | no |
| d0072 | pure_sky | afternoon | 1.62 | 0.010 | 0.707 | 41.2 | yes |
| d0073 | coastal | dusk | 1.71 | 0.021 | 0.847 | 13.4 | yes |
| d0074 | urban | night | 1.63 | 0.042 | 0.813 | 27.6 | yes |
| d0075 | rural | dawn | 1.43 | 0.011 | 0.811 | 8.6 | yes |
| d0076 | forest | morning | 0.32 | 0.027 | 0.894 | 34.8 | yes |
| d0077 | mountain | afternoon | 1.43 | 0.050 | 0.535 | 27.6 | yes |
| d0078 | desert | dusk | 0.86 | 0.015 | 0.651 | 43.6 | yes |
| d0079 | industrial | night | 1.07 | 0.033 | 0.631 | 8.2 | yes |
| d0080 | airfield | dawn | 1.20 | 0.010 | 0.735 | 9.2 | yes |
| d0081 | pure_sky | morning | 1.27 | 0.021 | 0.913 | 26.1 | yes |
| d0082 | coastal | afternoon | 1.56 | 0.042 | 0.591 | 20.4 | yes |
| d0083 | urban | dusk | 1.91 | 0.011 | 0.916 | 30.8 | yes |
| d0084 | rural | night | 0.79 | 0.027 | 0.918 | -1.8 | yes |
| d0085 | forest | dawn | 1.92 | 0.050 | 0.571 | 14.7 | yes |
| d0086 | mountain | morning | 0.62 | 0.015 | 0.757 | 3.2 | yes |
| d0087 | desert | afternoon | 0.89 | 0.033 | 0.569 | 26.0 | yes |
| d0088 | industrial | dusk | 1.29 | 0.010 | 0.813 | 7.9 | yes |
| d0089 | airfield | night | 1.74 | 0.021 | 0.664 | 10.5 | yes |
| d0090 | pure_sky | dawn | 0.57 | 0.042 | 0.636 | -3.2 | yes |
| d0091 | coastal | morning | 1.08 | 0.011 | 0.806 | 10.6 | yes |
| d0092 | urban | afternoon | 1.10 | 0.027 | 0.880 | 3.8 | yes |
| d0093 | rural | dusk | 2.11 | 0.150 | 0.619 | 28.7 | yes |
| d0094 | forest | night | 1.62 | 0.015 | 0.704 | 18.5 | yes |
| d0095 | mountain | dawn | 1.16 | 0.033 | 0.607 | 1.3 | yes |
| d0096 | desert | morning | 0.99 | 0.010 | 0.656 | 22.2 | yes |
| d0097 | industrial | afternoon | 0.86 | 0.021 | 0.925 | 49.1 | no |
| d0098 | airfield | dusk | 0.99 | 0.042 | 0.741 | 4.1 | yes |
| d0099 | pure_sky | night | 1.33 | 0.011 | 0.784 | 61.6 | no |
| d0100 | coastal | dawn | 1.05 | 0.027 | 0.756 | 5.8 | yes |
| d0101 | urban | morning | 1.94 | 0.050 | 0.662 | 12.1 | yes |
| d0102 | rural | afternoon | 0.76 | 0.015 | 0.470 | 17.4 | yes |
| d0103 | forest | dusk | 1.19 | 0.033 | 0.559 | 17.0 | yes |
| d0104 | mountain | night | 1.49 | 0.010 | 0.417 | -3.8 | yes |
| d0105 | desert | dawn | 1.49 | 0.021 | 0.868 | 16.4 | yes |
| d0106 | industrial | morning | 1.83 | 0.042 | 0.418 | 9.2 | yes |
| d0107 | airfield | afternoon | 1.47 | 0.011 | 0.688 | 17.7 | yes |
| d0108 | pure_sky | dusk | 1.50 | 0.027 | 0.597 | 54.3 | no |
| d0109 | coastal | night | 1.23 | 0.050 | 0.499 | 15.7 | yes |
| d0110 | urban | dawn | 0.71 | 0.015 | 0.515 | 19.7 | yes |
| d0111 | rural | morning | 1.17 | 0.033 | 0.696 | 21.5 | yes |
| d0112 | forest | afternoon | 1.36 | 0.010 | 0.690 | -11.2 | yes |
| d0113 | mountain | dusk | 1.03 | 0.042 | 0.859 | -3.9 | yes |
| d0114 | desert | night | 0.95 | 0.042 | 0.515 | 59.2 | no |
| d0115 | industrial | dawn | 1.85 | 0.011 | 0.465 | 24.7 | yes |
| d0116 | airfield | morning | 1.14 | 0.027 | 0.515 | 68.9 | no |
| d0117 | pure_sky | afternoon | 0.80 | 0.050 | 0.377 | 8.3 | yes |
| d0118 | coastal | dusk | 0.89 | 0.015 | 0.571 | 23.7 | yes |
| d0119 | urban | night | 1.25 | 0.033 | 0.817 | 5.1 | yes |
| d0120 | rural | dawn | 1.65 | 0.010 | 0.741 | 37.0 | yes |
| d0121 | forest | morning | 5.71 | 0.208 | 0.500 | 11.8 | yes |
| d0122 | mountain | afternoon | 1.38 | 0.042 | 0.539 | 58.8 | no |
| d0123 | desert | dusk | 0.61 | 0.011 | 0.812 | 91.1 | no |
| d0124 | industrial | night | 0.82 | 0.027 | 0.728 | 25.3 | yes |
| d0125 | airfield | dawn | 1.38 | 0.050 | 0.815 | 5.6 | yes |
| d0126 | pure_sky | morning | 1.00 | 0.015 | 0.777 | 29.5 | yes |
| d0127 | coastal | afternoon | 1.22 | 0.033 | 0.725 | 5.4 | yes |
| d0128 | urban | dusk | 1.05 | 0.010 | 0.891 | 21.8 | yes |
| d0129 | rural | night | 0.83 | 0.021 | 0.726 | 11.5 | yes |
| d0130 | forest | dawn | 1.63 | 0.042 | 0.789 | 8.1 | yes |
| d0131 | mountain | morning | 1.36 | 0.011 | 0.498 | 11.8 | yes |
| d0132 | desert | afternoon | 1.22 | 0.027 | 0.500 | 17.0 | yes |
| d0133 | industrial | dusk | 0.89 | 0.050 | 0.648 | 39.4 | yes |
| d0134 | airfield | night | 0.82 | 0.015 | 0.621 | 9.0 | yes |
| d0135 | pure_sky | dawn | 1.59 | 0.033 | 0.734 | 43.1 | yes |
| d0136 | coastal | morning | 0.82 | 0.010 | 0.874 | 9.2 | yes |
| d0137 | urban | afternoon | 1.46 | 0.021 | 0.783 | 11.1 | yes |
| d0138 | rural | dusk | 0.52 | 0.042 | 0.559 | 9.7 | yes |
| d0139 | forest | night | 1.46 | 0.011 | 0.703 | 18.5 | yes |
| d0140 | mountain | dawn | 0.94 | 0.027 | 0.684 | 0.7 | yes |
| d0141 | desert | morning | 1.34 | 0.050 | 0.799 | -7.9 | yes |
| d0142 | industrial | afternoon | 0.84 | 0.015 | 0.803 | 6.9 | yes |
| d0143 | airfield | dusk | 1.86 | 0.033 | 0.822 | -1.3 | yes |
| d0144 | pure_sky | night | 1.57 | 0.010 | 0.709 | 90.0 | no |
| d0145 | coastal | dawn | 0.70 | 0.021 | 0.650 | 16.8 | yes |
| d0146 | urban | morning | 1.02 | 0.042 | 0.767 | 38.7 | yes |
| d0147 | rural | afternoon | 1.21 | 0.011 | 0.780 | 43.4 | yes |
| d0148 | forest | dusk | 1.13 | 0.027 | 0.571 | 12.2 | yes |
| d0149 | mountain | night | 1.28 | 0.050 | 0.713 | -3.4 | yes |
| d0150 | desert | dawn | 1.17 | 0.015 | 0.470 | -3.8 | yes |
| d0151 | industrial | morning | 1.03 | 0.033 | 0.810 | 15.3 | yes |
| d0152 | airfield | afternoon | 0.86 | 0.010 | 0.834 | 18.6 | yes |
| d0153 | pure_sky | dusk | 1.68 | 0.021 | 0.768 | 13.5 | yes |
| d0154 | coastal | night | 1.45 | 0.042 | 0.669 | 14.5 | yes |
| d0155 | urban | dawn | 0.89 | 0.011 | 0.938 | 23.5 | yes |
| d0156 | rural | morning | 1.04 | 0.027 | 0.596 | 3.8 | yes |
| d0157 | forest | afternoon | 1.33 | 0.050 | 0.451 | 13.6 | yes |
| d0158 | mountain | dusk | 0.95 | 0.015 | 0.787 | 10.0 | yes |
| d0159 | desert | night | 1.01 | 0.033 | 0.732 | 9.3 | yes |
| d0160 | industrial | dawn | 1.38 | 0.010 | 0.651 | 17.5 | yes |
| d0161 | airfield | morning | 1.38 | 0.021 | 0.711 | 10.8 | yes |
| d0162 | pure_sky | afternoon | 1.29 | 0.042 | 0.723 | 43.1 | yes |
| d0163 | coastal | dusk | 0.66 | 0.011 | 0.834 | 64.4 | no |
| d0164 | urban | night | 0.75 | 0.027 | 0.760 | 16.2 | yes |
| d0165 | rural | dawn | 0.97 | 0.050 | 0.748 | 47.2 | no |
| d0166 | forest | morning | 1.18 | 0.015 | 0.803 | 13.2 | yes |
| d0167 | mountain | afternoon | 1.35 | 0.067 | 0.653 | 10.4 | yes |
| d0168 | desert | dusk | 1.28 | 0.010 | 0.767 | -2.1 | yes |
| d0169 | industrial | night | 1.64 | 0.021 | 0.617 | 5.4 | yes |
| d0170 | airfield | dawn | 1.23 | 0.042 | 0.537 | 12.3 | yes |
| d0171 | pure_sky | morning | 1.12 | 0.011 | 0.655 | 93.7 | no |
| d0172 | coastal | afternoon | 0.62 | 0.027 | 0.682 | 20.1 | yes |
| d0173 | urban | dusk | 1.54 | 0.050 | 0.528 | 14.7 | yes |
| d0174 | rural | night | 1.22 | 0.015 | 0.709 | 41.7 | yes |
| d0175 | forest | dawn | 1.59 | 0.033 | 0.818 | 26.4 | yes |
| d0176 | mountain | morning | 1.27 | 0.010 | 0.724 | 56.1 | no |
| d0177 | desert | afternoon | 1.45 | 0.021 | 0.626 | 19.0 | yes |
| d0178 | industrial | dusk | 1.49 | 0.042 | 0.524 | 44.0 | yes |
| d0179 | airfield | night | 1.35 | 0.011 | 0.954 | 34.8 | yes |
| d0180 | pure_sky | dawn | 1.02 | 0.027 | 0.763 | 52.2 | no |
| d0181 | coastal | morning | 1.00 | 0.050 | 0.796 | 86.0 | no |
| d0182 | urban | afternoon | 1.10 | 0.015 | 0.711 | 28.9 | yes |
| d0183 | rural | dusk | 1.07 | 0.033 | 0.771 | 2.5 | yes |
| d0184 | forest | night | 1.32 | 0.010 | 0.833 | 17.0 | yes |
| d0185 | mountain | dawn | 1.03 | 0.021 | 0.707 | 6.9 | yes |
| d0186 | desert | morning | 1.57 | 0.042 | 0.704 | 6.1 | yes |
| d0187 | industrial | afternoon | 1.97 | 0.011 | 0.789 | 19.0 | yes |
| d0188 | airfield | dusk | 0.82 | 0.027 | 0.601 | 7.6 | yes |
| d0189 | pure_sky | night | 1.82 | 0.050 | 0.651 | 26.8 | yes |
| d0190 | coastal | dawn | 0.76 | 0.015 | 0.576 | 9.7 | yes |
| d0191 | urban | morning | 1.10 | 0.033 | 0.565 | 14.9 | yes |
| d0192 | rural | afternoon | 1.24 | 0.010 | 0.917 | 10.1 | yes |
| d0193 | forest | dusk | 1.28 | 0.021 | 0.844 | 19.4 | yes |
| d0194 | mountain | night | 1.19 | 0.042 | 0.611 | 4.4 | yes |
| d0195 | desert | dawn | 1.04 | 0.011 | 0.805 | -8.4 | yes |
| d0196 | industrial | morning | 0.82 | 0.027 | 0.839 | 62.2 | no |
| d0197 | airfield | afternoon | 1.40 | 0.050 | 0.676 | 17.4 | yes |
| d0198 | pure_sky | dusk | 1.69 | 0.031 | 0.810 | 0.6 | yes |
| d0199 | coastal | night | 1.21 | 0.033 | 0.628 | 29.8 | yes |
| d0200 | urban | dawn | 1.20 | 0.010 | 0.571 | 8.8 | yes |
| d0201 | rural | morning | 1.04 | 0.021 | 0.924 | 13.7 | yes |
| d0202 | forest | afternoon | 1.52 | 0.042 | 0.750 | 5.5 | yes |
| d0203 | mountain | dusk | 1.59 | 0.011 | 0.594 | 11.4 | yes |
| d0204 | desert | night | 0.78 | 0.027 | 0.708 | 32.4 | yes |
| d0205 | industrial | dawn | 1.49 | 0.050 | 0.708 | 16.1 | yes |
| d0206 | airfield | morning | 0.85 | 0.015 | 0.901 | 16.8 | yes |
| d0207 | pure_sky | afternoon | 1.05 | 0.033 | 0.862 | 4.4 | yes |
| d0208 | coastal | dusk | 0.74 | 0.010 | 0.637 | 47.5 | no |
| d0209 | urban | night | 0.68 | 0.021 | 0.774 | 82.1 | no |
| d0210 | rural | dawn | 1.19 | 0.042 | 0.536 | 17.8 | yes |
| d0211 | forest | morning | 1.62 | 0.011 | 0.577 | 14.6 | yes |
| d0212 | mountain | afternoon | 1.15 | 0.027 | 0.623 | 12.9 | yes |
| d0213 | desert | dusk | 0.86 | 0.050 | 0.736 | -3.7 | yes |
| d0214 | industrial | night | 1.00 | 0.015 | 0.788 | 32.5 | yes |
| d0215 | airfield | dawn | 1.08 | 0.033 | 0.786 | 12.8 | yes |
| d0216 | pure_sky | morning | 0.73 | 0.010 | 0.804 | 85.8 | no |
| d0217 | coastal | afternoon | 1.00 | 0.021 | 0.564 | -0.5 | yes |
| d0218 | urban | dusk | 0.76 | 0.042 | 0.637 | 57.5 | no |
| d0219 | rural | night | 1.26 | 0.011 | 0.564 | 15.8 | yes |
| d0220 | forest | dawn | 0.97 | 0.027 | 0.684 | 10.9 | yes |
| d0221 | mountain | morning | 1.32 | 0.050 | 0.759 | 40.5 | yes |
| d0222 | desert | afternoon | 0.45 | 0.015 | 0.545 | -0.1 | yes |
| d0223 | industrial | dusk | 1.76 | 0.033 | 0.522 | 12.2 | yes |
| d0224 | airfield | night | 1.33 | 0.010 | 0.752 | 9.1 | yes |
| d0225 | pure_sky | dawn | 0.72 | 0.021 | 0.769 | 23.1 | yes |
| d0226 | coastal | morning | 1.57 | 0.042 | 0.529 | 12.1 | yes |
| d0227 | urban | afternoon | 1.67 | 0.011 | 0.744 | -0.3 | yes |
| d0228 | rural | dusk | 1.02 | 0.027 | 0.710 | 30.4 | yes |
| d0229 | forest | night | 1.57 | 0.050 | 0.656 | 10.6 | yes |
| d0230 | mountain | dawn | 0.57 | 0.015 | 0.848 | 58.2 | no |
| d0231 | desert | morning | 1.63 | 0.033 | 0.735 | 47.5 | no |
| d0232 | industrial | afternoon | 0.94 | 0.010 | 0.770 | 7.7 | yes |
| d0233 | airfield | dusk | 1.21 | 0.021 | 0.726 | 0.8 | yes |
| d0234 | pure_sky | night | 0.51 | 0.083 | 0.764 | 26.9 | yes |
| d0235 | coastal | dawn | 1.04 | 0.011 | 0.545 | 10.0 | yes |
| d0236 | urban | morning | 1.03 | 0.027 | 0.649 | 23.0 | yes |
| d0237 | rural | afternoon | 0.82 | 0.050 | 0.804 | 4.5 | yes |
| d0238 | forest | dusk | 1.23 | 0.015 | 0.588 | 3.5 | yes |
| d0239 | mountain | night | 1.03 | 0.033 | 0.668 | -6.2 | yes |
| d0240 | desert | dawn | 1.39 | 0.010 | 0.848 | -0.1 | yes |
| d0241 | industrial | morning | 1.19 | 0.021 | 0.683 | 1.2 | yes |
| d0242 | airfield | afternoon | 3.04 | 0.125 | 0.675 | 8.3 | yes |
| d0243 | pure_sky | dusk | 1.42 | 0.011 | 0.693 | 2.6 | yes |
| d0244 | coastal | night | 0.88 | 0.027 | 0.710 | 3.2 | yes |
| d0245 | urban | dawn | 1.48 | 0.050 | 0.525 | 23.0 | yes |
| d0246 | rural | morning | 0.66 | 0.015 | 0.725 | 18.8 | yes |
| d0247 | forest | afternoon | 1.59 | 0.033 | 0.551 | 36.5 | yes |
| d0248 | mountain | dusk | 1.67 | 0.010 | 0.649 | 16.2 | yes |
| d0249 | desert | night | 1.49 | 0.021 | 0.863 | -2.6 | yes |
| d0250 | industrial | dawn | 0.87 | 0.042 | 0.836 | 78.5 | no |
| d0251 | airfield | morning | 1.03 | 0.011 | 0.886 | 67.3 | no |
| d0252 | pure_sky | afternoon | 1.59 | 0.027 | 0.602 | 55.6 | no |
| d0253 | coastal | dusk | 1.10 | 0.050 | 0.753 | 34.3 | yes |
| d0254 | urban | night | 0.78 | 0.015 | 0.727 | 36.3 | yes |
| d0255 | rural | dawn | 1.41 | 0.033 | 0.440 | 4.6 | yes |
| d0256 | forest | morning | 0.89 | 0.010 | 0.760 | 25.0 | yes |
| d0257 | mountain | afternoon | 1.57 | 0.021 | 0.824 | 0.0 | yes |
| d0258 | desert | dusk | 1.21 | 0.042 | 0.705 | 6.9 | yes |
| d0259 | industrial | night | 0.96 | 0.011 | 0.808 | 47.0 | no |
| d0260 | airfield | dawn | 1.04 | 0.027 | 0.699 | 11.4 | yes |
| d0261 | pure_sky | morning | 0.84 | 0.050 | 0.425 | 33.3 | yes |
| d0262 | coastal | afternoon | 1.17 | 0.015 | 0.621 | 13.9 | yes |
| d0263 | urban | dusk | 0.90 | 0.033 | 0.671 | 38.8 | yes |
| d0264 | rural | night | 1.34 | 0.010 | 0.752 | 43.8 | yes |
| d0265 | forest | dawn | 1.22 | 0.021 | 0.680 | 15.6 | yes |
| d0266 | mountain | morning | 1.33 | 0.042 | 0.684 | 3.8 | yes |
| d0267 | desert | afternoon | 0.88 | 0.011 | 0.669 | 3.8 | yes |
| d0268 | industrial | dusk | 1.37 | 0.027 | 0.604 | 9.4 | yes |
| d0269 | airfield | night | 1.62 | 0.050 | 0.612 | 24.2 | yes |
| d0270 | pure_sky | dawn | 0.68 | 0.015 | 0.909 | 12.2 | yes |
| d0271 | coastal | morning | 1.68 | 0.033 | 0.835 | 8.1 | yes |
| d0272 | urban | afternoon | 1.55 | 0.010 | 0.900 | 44.0 | yes |
| d0273 | rural | dusk | 1.64 | 0.021 | 0.689 | 4.0 | yes |
| d0274 | forest | night | 1.25 | 0.042 | 0.660 | -3.4 | yes |
| d0275 | mountain | dawn | 1.38 | 0.011 | 0.886 | -2.4 | yes |
| d0276 | desert | morning | 1.08 | 0.027 | 0.552 | 4.2 | yes |
| d0277 | industrial | afternoon | 1.48 | 0.050 | 0.512 | 9.3 | yes |
| d0278 | airfield | dusk | 0.71 | 0.015 | 0.788 | 18.5 | yes |
| d0279 | pure_sky | night | 1.37 | 0.033 | 0.553 | 46.9 | no |
| d0280 | coastal | dawn | 1.12 | 0.010 | 0.816 | 44.8 | yes |
| d0281 | urban | morning | 1.12 | 0.021 | 0.622 | -3.8 | yes |
| d0282 | rural | afternoon | 1.32 | 0.042 | 0.612 | 11.3 | yes |
| d0283 | forest | dusk | 1.15 | 0.011 | 0.766 | 14.5 | yes |
| d0284 | mountain | night | 0.40 | 0.027 | 0.841 | 29.4 | yes |
| d0285 | desert | dawn | 1.89 | 0.050 | 0.582 | 10.2 | yes |
| d0286 | industrial | morning | 1.12 | 0.015 | 0.906 | 12.6 | yes |
| d0287 | airfield | afternoon | 1.65 | 0.033 | 0.769 | 21.4 | yes |
| d0288 | pure_sky | dusk | 0.69 | 0.010 | 0.781 | 12.9 | yes |
| d0289 | coastal | night | 1.30 | 0.021 | 0.740 | 36.3 | yes |
| d0290 | urban | dawn | 1.75 | 0.042 | 0.569 | 38.6 | yes |
| d0291 | rural | morning | 0.92 | 0.011 | 0.959 | 4.1 | yes |
| d0292 | forest | afternoon | 0.94 | 0.027 | 0.891 | 14.4 | yes |
| d0293 | mountain | dusk | 1.72 | 0.050 | 0.666 | 31.5 | yes |
| d0294 | desert | night | 1.13 | 0.015 | 0.795 | 33.3 | yes |
| d0295 | industrial | dawn | 1.43 | 0.033 | 0.671 | -7.5 | yes |
| d0296 | airfield | morning | 1.62 | 0.010 | 0.602 | -1.5 | yes |
| d0297 | pure_sky | afternoon | 1.45 | 0.021 | 0.859 | 15.8 | yes |
| d0298 | coastal | dusk | 0.90 | 0.042 | 0.699 | 4.6 | yes |
| d0299 | urban | night | 1.90 | 0.011 | 0.788 | 8.0 | yes |
