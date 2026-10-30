# Module M1 semantic-parser benchmark

Benchmark: 100 manually annotated descriptions in 6 subsets. Model: `deepseek-chat`, temperature 0.1.

## Headline

| Metric | LLM parser | Deterministic template parser |
|---|---:|---:|
| Field-level accuracy | 0.985 | 0.850 |
| Complete-schema match rate | 0.930 | 0.570 |
| Invalid-output rate | 0.000 | 0.000 |
| Mean latency (s) | 1.10 | 0.0001 |

## Field-level accuracy

| Field | LLM | Template | n |
|---|---:|---:|---:|
| scene_type | 0.979 | 0.821 | 95 |
| time_of_day | 1.000 | 0.804 | 92 |
| weather | 0.967 | 0.868 | 91 |
| camera_position | 1.000 | 0.862 | 94 |
| action | 0.979 | 0.895 | 95 |

## By subset

| Subset | n | LLM joint | Template joint | LLM complete |
|---|---:|---:|---:|---:|
| canonical_cn | 20 | 1.000 | 0.970 | 1.000 |
| canonical_en | 20 | 1.000 | 0.900 | 1.000 |
| conflict_oov | 10 | 0.765 | 0.941 | 0.600 |
| paraphrase_cn | 20 | 0.980 | 0.760 | 0.900 |
| paraphrase_en | 15 | 0.987 | 0.587 | 0.933 |
| underspecified | 15 | 1.000 | 0.987 | 1.000 |

## Trajectory extraction

| Arm | distance-first | distance-last | both correct | n |
|---|---:|---:|---:|---:|
| llm | 1.000 | 1.000 | 1.000 | 9 |
| template | 1.000 | 1.000 | 1.000 | 9 |

## Ambiguity handling (underspecified + out-of-vocabulary)

| Arm | any note on ambiguous | note naming the unresolved dimension | note on determinate |
|---|---:|---:|---:|
| llm | 1.000 | 1.000 | 1.000 |
| template | 0.600 | n/a | 0.000 |

## Invalid-output repair (analysis arm only)

| Quantity | Value |
|---|---:|
| Invalid outputs, this pass | 0 / 100 |
| Recovered by brace balancing | 0 |
| Invalid-output rate after repair | 0.000 |
| Joint accuracy after repair | 0.985 |

## Replication across passes (decoder sampling variance)

| Quantity | Value |
|---|---:|
| Passes completed | 3 |
| Pooled completions | 300 |
| Pooled invalid-output rate | 0.0000 |
| Joint accuracy, mean +/- sd | 0.985 +/- 0.000 |
| Complete-schema match, mean +/- sd | 0.930 +/- 0.000 |

## Caveats

1. The out-of-vocabulary subset defines its gold standard as the conservative fallback value, which structurally favours the template arm: an arm that always returns the default is always 'correct' there. The informative quantity for that subset is therefore not accuracy but whether the fallback is signalled.
2. The template arm is a longest-match keyword parser over the *same* controlled vocabulary as the deployed system, so it shares the deployed decoder rather than representing an independently engineered baseline.
3. Latency excludes network variance in the template arm, which is local and network-free by construction.

## Failure modes (as deployed)


