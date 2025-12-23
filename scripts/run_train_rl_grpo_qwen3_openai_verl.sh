### Running Mac

## Prepare Verl Datasets, output
## train.parquet files
python train_rl_grpo_qwen3_openai_verl.py
## verl_train.parquet, "../train/dataset/rl/verl_train.parquet"

## Model Cache Found From HF or Dashscope

## Prepare Verl Datasets and start Rays
python -m verl.trainer.main_ppo \
    data.train_files="../train/dataset/rl/verl_train.parquet" \
    data.val_files="../train/dataset/rl/verl_train.parquet" \
    data.train_batch_size=1 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="../train/model/qwen3/Qwen3-0.6B/Qwen/Qwen3-0___6B" \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.name=hf \
    trainer.n_gpus_per_node=0 \
    trainer.nnodes=1 \
    trainer.project_name="grpo_qwen3_rl_verl"


```
2025-12-23 18:36:28,398	INFO worker.py:1942 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265 
(pid=9625) /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/verl/__init__.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
(pid=9625)   import pkg_resources

```

