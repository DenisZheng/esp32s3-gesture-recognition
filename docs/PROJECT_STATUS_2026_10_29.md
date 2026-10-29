# 项目当前状态（2026-10-29）

本文件记录截至 `2026-10-29 10:07`（服务器本地时间 UTC+8）的真实状态，
用于交接和后续恢复。所有结论均有服务器上的文件或日志对应，不是计划值。

## 1. 代码基线

```text
仓库    : github.com/DenisZheng/esp32s3-gesture-recognition
main    : fae3172
CI      : run #159 success
```

`main` 已包含：

- `--full` 全量数据集构建模式（仅 `local-dir`）
- `p1_qa_server.py` 回环地址浏览器抽检入口
- `p1_dataset_qa accept` 研究用途验收
- `docs/P2_HAGRIDV2_FULL_TRAINING.md` 全量训练过程记录

## 2. 十类模型训练结果（已终止）

训练在 **epoch 62** 被 Linux OOM killer 终止，不是正常结束。

```text
终止时间   : 2026-10-28T22:15:23+08:00
内核证据   : Out of memory: Killed process 103297 (python)
             total-vm: 52,111,888 kB  anon-rss: 27,695,248 kB
主机内存   : 31 GiB
最后完成   : epoch 61
```

epoch 61 指标：

| 指标 | 值 |
|---|---:|
| mAP50 | **0.96746** |
| mAP50-95 | **0.74482** |

对照：此前 6000 张数据集的模型为 mAP50 `0.724` / mAP50-95 `0.456`。

### 崩溃前的劣化特征

```text
前期      : 约 46.8 it/s（约 10–12 分钟/epoch）
epoch 62  : 约 1.27 it/s（19–22 秒/step），慢约 37 倍
```

原因是内存压力导致 swap thrashing，dataloader 喂不上数据，不是 GPU 算力不足。
崩溃前 6 小时检查时 swap 已用 `7.6 / 8 GiB`，当时被误判为"偏高但稳定"。

## 3. 当前保留的资产

### 模型权重

```text
runs/p2-full-v2-200-r2/weights/best.pt   1,855,760 bytes
runs/p2-full-v2-200-r2/weights/last.pt   1,855,760 bytes
SHA256 (both) : fd8ed094d22bdc82...
```

两者 SHA256 相同，说明 epoch 61 即为最后一个完成 epoch 的最佳点。

### 训练证据

```text
runs/p2-full-v2-200-r2/results.csv             逐 epoch 指标
runs/p2-full-v2-200-r2/p2_terminated_oom.json  终止原因、时间、权重 SHA256
runs/p2-full-v2-200-r2/dataset_evidence/
    manifest.aggregate.json                    被删数据集的 manifest 快照
    v2_ingest_report.json
    dataset.yaml
data/manual/p2-full-v2-train-r2.log            完整训练日志
data/manual/p2-full-v2-train.log               第一次失败（Arial.ttf）日志
```

### 数据

```text
data/manual/hagrid_annotations/    981 MB    30 个 JSON（train/val/test × 10 类）
```

## 4. 已删除的资产

| 资产 | 大小 | 删除时间 | 说明 |
|---|---:|---|---|
| `data/processed/hagrid10-v2-512` | 33 GB | 2026-10-29 10:01 | 全量构建数据集（281,369 张） |
| `data/manual/hagrid_dataset` | 32 GB | 2026-10-29 10:07 | 十类原始图片（281,795 张） |
| `data/cache/hagrid10-p1-05/images` | 14 GB | 2026-10-28 14:21 | 旧缓存图片 |
| `data/processed/hagrid10-500-50-50-v1` | 7.2 GB | 2026-10-28 14:21 | 6000 张旧数据集 |
| P1-06 旧审核图片包 | 约 110 MB | 2026-10-28 14:21 | 保留 JSON/manifest |
| `data/reviews/hagrid-v2-full-qa/images` | 见下 | 随数据集删除 | QA 抽检图副本 |

根分区可用空间：**179 GB**。

## 5. 重要约束：`best.pt` 目前无法复评

`best.pt` 存在，但**没有对应数据集**：

- 构建好的数据集已删
- 原始十类图片已删
- 移动硬盘（`/dev/sdb1`）已卸载

若要复评该权重（test 评估、ONNX 导出验证、量化校准），必须先：

1. 挂载移动硬盘：`udisksctl mount -b /dev/sdb1`（需在图形会话 D-Bus 下执行）
2. 重新复制十类图片到 `data/manual/hagrid_dataset/`
3. 用 `--full` 重建数据集
4. 才能运行 `p2_train eval` / `p3_eval`

标签 JSON 已保留，重建不需要重新获取标注。

## 6. 未完成项

- 200 epoch 训练未完成（停在 61）
- 独立 test 评估未做（`best.pt` 无数据集）
- ONNX 导出与 PyTorch/ONNX parity 未做
- ESP-PPQ W8A8 量化未做
- CoreS3 实机部署未做
- 数据集人工 QA 为 240 张分层抽样，不是 281,369 张逐张确认
- 商业使用与再分发许可仍未澄清

## 7. 已知缺陷

### OOM 与内存监控缺失

训练无内存上限、无监控告警。`workers=4` 在 208k 样本数据集上长期运行后
单进程涨到 27.7 GB。后续训练应：

- 降低 `workers`（2 或更少）
- 监控 swap 使用率，超过 50% 即告警
- 考虑周期性重启 dataloader worker

### `p2_train` 不支持 `--resume`

被 OOM 杀掉后无法从 `last.pt` 续训，直接重跑会从头开始。
`last.pt` 与 `best.pt` 相同，因为 epoch 62 未完成。

### Windows 测试环境

本地 Windows 上 29 个测试因平台差异失败（`os.fchmod`、GBK 默认编码、
符号链接权限、路径分隔符）。Linux CI 全绿，这些不是真实缺陷。

## 8. 恢复检查清单

新会话接手时按顺序确认：

```bash
# 1. 代码基线
git -C ~/esp32s3-gesture-recognition log --oneline -1     # 期望 fae3172
git -C ~/esp32s3-gesture-recognition status --short

# 2. 保留资产仍在
ls -l ~/esp32s3-gesture-recognition-full/runs/p2-full-v2-200-r2/weights/
ls -l ~/esp32s3-gesture-recognition-full/runs/p2-full-v2-200-r2/p2_terminated_oom.json
find ~/esp32s3-gesture-recognition/data/manual/hagrid_annotations -name '*.json' | wc -l

# 3. 空间
df -h /

# 4. 是否有残留训练进程
tmux ls
docker ps
```

## 9. 下一个目标：官方 HaGRID 手势模型训练

已确认放弃当前十类删减数据集，转向官方 HaGRID 模型。开始前需要明确：

- 官方数据集的获取方式与存放路径
- 目标类别数与 ESP-DL 参考手势的映射关系
- 输入分辨率是否仍为 `224×224`
- 是否复用现有 `espdet-gesture:torch220-cu121` 容器

这些尚未开始，本文件不预设结论。
