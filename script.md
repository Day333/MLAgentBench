# 数据集下载

## 需要下载/准备数据的任务

```cmd
python -u -m MLAgentBench.prepare_task amp-parkinsons-disease-progression-prediction
python -u -m MLAgentBench.prepare_task babylm
python -u -m MLAgentBench.prepare_task cifar10
python -u -m MLAgentBench.prepare_task fathomnet
python -u -m MLAgentBench.prepare_task feedback
python -u -m MLAgentBench.prepare_task house-price
python -u -m MLAgentBench.prepare_task identify-contrails
python -u -m MLAgentBench.prepare_task ogbn-arxiv
python -u -m MLAgentBench.prepare_task spaceship-titanic
```

## 不需要额外准备(数据已在 env/ 目录里)的任务

这些任务跑 `prepare_task` 也没关系(会提示 `xxx dataset needs no preparation`),不跑也行:

```cmd
python -u -m MLAgentBench.prepare_task CLRS
python -u -m MLAgentBench.prepare_task bibtex-generation
python -u -m MLAgentBench.prepare_task imdb
python -u -m MLAgentBench.prepare_task literature-review-tool
python -u -m MLAgentBench.prepare_task llama-inference
python -u -m MLAgentBench.prepare_task vectorization
```


# quick start

所有运行日志统一放在根目录的 `log/` 文件夹下,每个任务/agent 组合各占一个子目录(`log/<task>-baseline`、`log/<task>-ra`)。

## amp-parkinsons-disease-progression-prediction

- baseline

```cmd
# 训练
mkdir -p log/amp-parkinsons-disease-progression-prediction-baseline && python -u -m MLAgentBench.runner --task amp-parkinsons-disease-progression-prediction --device 0 --log-dir log/amp-parkinsons-disease-progression-prediction-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/amp-parkinsons-disease-progression-prediction-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/amp-parkinsons-disease-progression-prediction-baseline --task amp-parkinsons-disease-progression-prediction --output-file log/amp-parkinsons-disease-progression-prediction-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/amp-parkinsons-disease-progression-prediction-ra && python -u -m MLAgentBench.runner --task amp-parkinsons-disease-progression-prediction --device 0 --log-dir log/amp-parkinsons-disease-progression-prediction-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/amp-parkinsons-disease-progression-prediction-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/amp-parkinsons-disease-progression-prediction-ra --task amp-parkinsons-disease-progression-prediction --output-file log/amp-parkinsons-disease-progression-prediction-ra/results.json
```

## babylm

- baseline

```cmd
# 训练
mkdir -p log/babylm-baseline && python -u -m MLAgentBench.runner --task babylm --device 0 --log-dir log/babylm-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/babylm-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/babylm-baseline --task babylm --output-file log/babylm-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/babylm-ra && python -u -m MLAgentBench.runner --task babylm --device 0 --log-dir log/babylm-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/babylm-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/babylm-ra --task babylm --output-file log/babylm-ra/results.json
```

## cifar10

- baseline

```cmd
# 训练
mkdir -p log/cifar10-baseline && python -u -m MLAgentBench.runner --task cifar10 --device 0 --log-dir log/cifar10-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/cifar10-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/cifar10-baseline --task cifar10 --output-file log/cifar10-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/cifar10-ra && python -u -m MLAgentBench.runner --task cifar10 --device 0 --log-dir log/cifar10-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/cifar10-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/cifar10-ra --task cifar10 --output-file log/cifar10-ra/results.json
```

## fathomnet

- baseline

```cmd
# 训练
mkdir -p log/fathomnet-baseline && python -u -m MLAgentBench.runner --task fathomnet --device 0 --log-dir log/fathomnet-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/fathomnet-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/fathomnet-baseline --task fathomnet --output-file log/fathomnet-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/fathomnet-ra && python -u -m MLAgentBench.runner --task fathomnet --device 0 --log-dir log/fathomnet-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/fathomnet-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/fathomnet-ra --task fathomnet --output-file log/fathomnet-ra/results.json
```

## feedback

- baseline

```cmd
# 训练
mkdir -p log/feedback-baseline && python -u -m MLAgentBench.runner --task feedback --device 0 --log-dir log/feedback-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/feedback-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/feedback-baseline --task feedback --output-file log/feedback-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/feedback-ra && python -u -m MLAgentBench.runner --task feedback --device 0 --log-dir log/feedback-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/feedback-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/feedback-ra --task feedback --output-file log/feedback-ra/results.json
```

## house-price

- baseline

```cmd
# 训练
mkdir -p log/house-price-baseline && python -u -m MLAgentBench.runner --task house-price --device 0 --log-dir log/house-price-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/house-price-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/house-price-baseline --task house-price --output-file log/house-price-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/house-price-ra && python -u -m MLAgentBench.runner --task house-price --device 0 --log-dir log/house-price-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/house-price-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/house-price-ra --task house-price --output-file log/house-price-ra/results.json
```

## identify-contrails

- baseline

```cmd
# 训练
mkdir -p log/identify-contrails-baseline && python -u -m MLAgentBench.runner --task identify-contrails --device 0 --log-dir log/identify-contrails-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/identify-contrails-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/identify-contrails-baseline --task identify-contrails --output-file log/identify-contrails-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/identify-contrails-ra && python -u -m MLAgentBench.runner --task identify-contrails --device 0 --log-dir log/identify-contrails-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/identify-contrails-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/identify-contrails-ra --task identify-contrails --output-file log/identify-contrails-ra/results.json
```

## ogbn-arxiv

- baseline

```cmd
# 训练
mkdir -p log/ogbn-arxiv-baseline && python -u -m MLAgentBench.runner --task ogbn-arxiv --device 0 --log-dir log/ogbn-arxiv-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/ogbn-arxiv-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/ogbn-arxiv-baseline --task ogbn-arxiv --output-file log/ogbn-arxiv-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/ogbn-arxiv-ra && python -u -m MLAgentBench.runner --task ogbn-arxiv --device 0 --log-dir log/ogbn-arxiv-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/ogbn-arxiv-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/ogbn-arxiv-ra --task ogbn-arxiv --output-file log/ogbn-arxiv-ra/results.json
```

## spaceship-titanic

- baseline

```cmd
# 训练
mkdir -p log/spaceship-titanic-baseline && python -u -m MLAgentBench.runner --task spaceship-titanic --device 0 --log-dir log/spaceship-titanic-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/spaceship-titanic-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/spaceship-titanic-baseline --task spaceship-titanic --output-file log/spaceship-titanic-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/spaceship-titanic-ra && python -u -m MLAgentBench.runner --task spaceship-titanic --device 0 --log-dir log/spaceship-titanic-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/spaceship-titanic-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/spaceship-titanic-ra --task spaceship-titanic --output-file log/spaceship-titanic-ra/results.json
```

## CLRS

- baseline

```cmd
# 训练
mkdir -p log/CLRS-baseline && python -u -m MLAgentBench.runner --task CLRS --device 0 --log-dir log/CLRS-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/CLRS-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/CLRS-baseline --task CLRS --output-file log/CLRS-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/CLRS-ra && python -u -m MLAgentBench.runner --task CLRS --device 0 --log-dir log/CLRS-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/CLRS-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/CLRS-ra --task CLRS --output-file log/CLRS-ra/results.json
```

## bibtex-generation

- baseline

```cmd
# 训练
mkdir -p log/bibtex-generation-baseline && python -u -m MLAgentBench.runner --task bibtex-generation --device 0 --log-dir log/bibtex-generation-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/bibtex-generation-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/bibtex-generation-baseline --task bibtex-generation --output-file log/bibtex-generation-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/bibtex-generation-ra && python -u -m MLAgentBench.runner --task bibtex-generation --device 0 --log-dir log/bibtex-generation-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/bibtex-generation-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/bibtex-generation-ra --task bibtex-generation --output-file log/bibtex-generation-ra/results.json
```

## imdb

- baseline

```cmd
# 训练
mkdir -p log/imdb-baseline && python -u -m MLAgentBench.runner --task imdb --device 0 --log-dir log/imdb-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/imdb-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/imdb-baseline --task imdb --output-file log/imdb-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/imdb-ra && python -u -m MLAgentBench.runner --task imdb --device 0 --log-dir log/imdb-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/imdb-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/imdb-ra --task imdb --output-file log/imdb-ra/results.json
```

## literature-review-tool

- baseline

```cmd
# 训练
mkdir -p log/literature-review-tool-baseline && python -u -m MLAgentBench.runner --task literature-review-tool --device 0 --log-dir log/literature-review-tool-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/literature-review-tool-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/literature-review-tool-baseline --task literature-review-tool --output-file log/literature-review-tool-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/literature-review-tool-ra && python -u -m MLAgentBench.runner --task literature-review-tool --device 0 --log-dir log/literature-review-tool-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/literature-review-tool-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/literature-review-tool-ra --task literature-review-tool --output-file log/literature-review-tool-ra/results.json
```

## llama-inference

- baseline

```cmd
# 训练
mkdir -p log/llama-inference-baseline && python -u -m MLAgentBench.runner --task llama-inference --device 0 --log-dir log/llama-inference-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/llama-inference-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/llama-inference-baseline --task llama-inference --output-file log/llama-inference-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/llama-inference-ra && python -u -m MLAgentBench.runner --task llama-inference --device 0 --log-dir log/llama-inference-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/llama-inference-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/llama-inference-ra --task llama-inference --output-file log/llama-inference-ra/results.json
```

## vectorization

- baseline

```cmd
# 训练
mkdir -p log/vectorization-baseline && python -u -m MLAgentBench.runner --task vectorization --device 0 --log-dir log/vectorization-baseline --work-dir workspace --agent-type Agent --python $(which python)  >  log/vectorization-baseline/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/vectorization-baseline --task vectorization --output-file log/vectorization-baseline/results.json
```

- ResearchAgent

```cmd
# 训练
mkdir -p log/vectorization-ra && python -u -m MLAgentBench.runner --task vectorization --device 0 --log-dir log/vectorization-ra --work-dir workspace --llm-name qwen3.7-flash --edit-script-llm-name qwen3.7-flash --fast-llm-name qwen3.7-flash --agent-type ResearchAgent --python $(which python)  >  log/vectorization-ra/log 2>&1

# 测评
python -m MLAgentBench.eval --log-folder log/vectorization-ra --task vectorization --output-file log/vectorization-ra/results.json
```

