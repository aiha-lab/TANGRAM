# KeyDiff use case

Source: [`vllm/v1/attention/compression/keydiff.py`](../../vllm/v1/attention/compression/keydiff.py) ·
Paper: [arXiv:2504.15364](https://arxiv.org/abs/2504.15364)

## Speedup

<p align="center">
  <img src="../assets/speedup/speedup_keydiff.png" alt="Tangram end-to-end speedup, KeyDiff scorer" width="100%"/>
</p>

## Accuracy

KeyDiff is the second column.

<p align="center">
  <img src="../assets/accuracy/accuracy_scbench.png" alt="SCBench accuracy comparison, Tangram vs PyTorch across four compression methods" width="100%"/>
</p>

## How to run

Speedup:

```bash
cd benchmarks/tangram/speedup
SCORERS=keydiff RATIOS="1.0 0.5 0.25 0.1" ./run_speedup.sh
```

SCBench accuracy:

```bash
cd benchmarks/tangram
SCORER=keydiff LEVEL=perlayer_cluster DATASET=mid RATIOS="1.0 0.5 0.25 0.1" \
bash benchmark_scbench.sh
```

Override `MODEL=` for another model.
