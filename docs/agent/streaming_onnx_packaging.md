# 流式 ONNX 打包

`dev` 分支把 FunASR 流式服务需要的模型导出成 ONNX，并发布到 GitHub Releases。`main` 只快进镜像 [modelscope/FunASR](https://github.com/modelscope/FunASR)，不存放这些脚本、规则或模型文件。

`upstream` 是原始仓库，`origin` 是 [apades/FunASR](https://github.com/apades/FunASR)。同步时从 `upstream` 拉取，不要把 `origin` 当成上游。

## 识别规则

槽位只来自 [runtime/run_server_2pass.sh](../../runtime/run_server_2pass.sh) 的 `model_dir`、`online_model_dir`、`vad_dir`、`punc_dir`。`itn_dir` 和 `lm_dir` 是 FST，不导出。

导出源按下面的顺序解析，并写进 `resolution`：

1. 与 `examples/industrial_data_pretraining/*/export.py` 里第一个 hub id 做名称对齐。比对前去掉组织名，并把 `-onnx` 与 `-pytorch` 看成同一个 checkpoint。必须整段名称相同，因此在线大模型不会误用示例里较小的非 large 模型。
2. 否则用 [funasr/download/name_maps_from_hub.py](../../funasr/download/name_maps_from_hub.py) 里 `name_maps_ms` 的别名。
3. 否则把组织改成 `iic`，并把 `-onnx` 换成 `-pytorch`。

导出调用与 [examples/industrial_data_pretraining/paraformer_streaming/export.py](../../examples/industrial_data_pretraining/paraformer_streaming/export.py) 相同：`AutoModel.export(type="onnx", quantize=False)`。不要改成 torchscript。

<!-- streaming-onnx-targets:start -->

### 当前目标

这一节由 `scripts/discover_streaming_onnx_targets.py` 生成，不要手改。

- **2pass 离线 ASR** (`model_dir` / `offline_asr`)
  - 运行时 ID：`damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx`
  - 导出 ID：`iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
  - 解析：`export_example`
  - 导出示例：`examples/industrial_data_pretraining/bicif_paraformer/export.py`
  - Hugging Face：无对应别名，下载失败时不会改走 HF
  - ONNX：`model.onnx`
  - 文件名来源：`funasr/models/bicif_paraformer/export_meta.py`

- **在线流式 ASR** (`online_model_dir` / `online_asr`)
  - 运行时 ID：`damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx`
  - 导出 ID：`iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online`
  - 解析：`name_map:paraformer-zh-streaming`
  - Hugging Face：`paraformer-zh-streaming` → `funasr/paraformer-zh-streaming`
  - ONNX：`model.onnx`、`decoder.onnx`
  - 文件名来源：`funasr/models/paraformer_streaming/export_meta.py`

- **VAD** (`vad_dir` / `vad`)
  - 运行时 ID：`damo/speech_fsmn_vad_zh-cn-16k-common-onnx`
  - 导出 ID：`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
  - 解析：`export_example`
  - 导出示例：`examples/industrial_data_pretraining/fsmn_vad_streaming/export.py`
  - Hugging Face：`fsmn-vad` → `funasr/fsmn-vad`
  - ONNX：`model.onnx`
  - 文件名来源：`funasr/models/fsmn_vad_streaming/export_meta.py`

- **实时标点** (`punc_dir` / `punc`)
  - 运行时 ID：`damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx`
  - 导出 ID：`iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727`
  - 解析：`export_example`
  - 导出示例：`examples/industrial_data_pretraining/ct_transformer_streaming/export.py`
  - Hugging Face：无对应别名，下载失败时不会改走 HF
  - ONNX：`model.onnx`
  - 文件名来源：`funasr/models/ct_transformer_streaming/export_meta.py`

<!-- streaming-onnx-targets:end -->

## 压缩包内容

每个槽位一个 zip。里面是导出的 ONNX，以及 checkpoint 里实际存在的随文件：`config.yaml`、`am.mvn`、`tokens.json`、`vad.yaml`、`vad.mvn`、`punc.yaml`、`seg_dict`。不打包 `.pt`、`.bin`、`.safetensors`。

## Release

工作流是 [.github/workflows/package-streaming-onnx.yml](../../.github/workflows/package-streaming-onnx.yml)。它只在 `dev` 被推送、且 diff 碰到识别脚本所读的路径、导出脚本或本文档时运行。

标签是 `streaming-onnx-<dev 短 SHA>`。说明里写该次 `dev` SHA 和当时的 `main` SHA。资产是四个 zip、`streaming_onnx_targets.json` 和 `SHA256SUMS`。

这份工作流不在默认分支 `main` 上，GitHub 网页里的手动运行入口不可用。要重新打包，把对应改动推到 `dev`。

`discover` 作业会跑 `python scripts/discover_streaming_onnx_targets.py --check`。清单和代码不一致时作业失败，不会发布模型。

## 失败时

GitHub 托管 runner 内存大约 7GB。Paraformer-large 导出如果内存不足，把仓库变量 `ONNX_RUNNER` 设成自有 runner 的标签。

从 runner 访问 modelscope.cn 可能超时。该槽位在 `name_maps_hf` 有别名时，导出脚本会用 `hub="hf"` 再试一次。没有别名时不会假装可以换源。

导出环境与仓库其他 CI 一样使用 CPU 版 `torch==2.10.0`，并安装 `onnx` 和 `onnxscript`。`torch.onnx.export` 会导入 `onnxscript`；缺少这个包时，权重可以加载，导出会立刻失败。

## 更新文档

改到 `runtime/run_server_2pass.sh`、`examples/industrial_data_pretraining/**/export.py`、`funasr/download/name_maps_from_hub.py` 或 `funasr/models/**/export_meta.py` 之后，在同一个 `dev` 提交里运行：

```shell
python scripts/discover_streaming_onnx_targets.py
python scripts/discover_streaming_onnx_targets.py --check
```

不要手改 `docs/agent/streaming_onnx_targets.json`，也不要手改上面的「当前目标」。
