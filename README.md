# Mu-scaling: Loss Prediction via Maximal Update Parametrization

We show that Maximal Update Parametrization (Mup) itself provides a model sequence that fits a modified scaling law and enables accurate loss prediction.

Mu-scaling paper: https://arxiv.org/abs/2304.06875

This implementation is based on [Huggingface](https://github.com/huggingface/transformers) and [MuTransformers](https://github.com/microsoft/mutransformers), with modifications to improve stability and support Deepspeed.

## Quick Start

1. Data Preparation

Preprocess datasets for causal language model following Huggingface [instructions](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling). We also provide an example of processed data in res/final_data/test.

2. Train GPT-2 with Mup

```bash
sh run_grid_search_pair_wise_mup.sh
```

3. Plot Loss Landscape

If Mup works correctly, loss basins for different widths should be aligned.

```python
python visualize_lr_landscape.py
```

4. Fit Scaling Laws

Record the training loss with the same data on the same step, then run

```python
python fit_scale_loss_prediction.py
```

5. Evaluation

If you would like to run on evaluation data, we suggest training all the models for more steps, and then

```bash
sh run_eval_ppl_loss_pred.sh
```

## References

If this project helps you, please cite us, thanks!
```
@article{DBLP:journals/corr/abs-2304-06875,
  author       = {Yiqun Yao and Yequan Wang},
  title        = {Research without Re-search: Maximal Update Parametrization Yields Accurate Loss Prediction across Scales},
  journal      = {CoRR},
  volume       = {abs/2304.06875},
  year         = {2023}
}
```