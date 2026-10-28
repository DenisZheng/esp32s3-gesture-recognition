# HaGRIDv2 512px 十类全量训练记录

本文记录 2026-10-27 至 2026-10-28 在 RTX 4080 主机上完成的 **HaGRIDv2 512px 十类全量**
数据准备与训练启动过程。所有数量、路径和状态均来自服务器上的实际运行日志与 manifest，
不是计划值。

> **范围声明**：此处的“全量”指项目 `hagrid10` 选定的十个 HaGRID source 类在各自
> `train/val/test` 原划分中的全部有效样本，**不是 HaGRID 的全部 34 个 source 类**。
>
> **许可声明**：该数据集与据此产出的权重仅用于**研究训练**，不得用于商业使用或商业分发。
> `legal_clearance_status = RESEARCH_ONLY_TRAINING_ONLY`，`commercial_use = false`。

## 1. 数据来源

移动硬盘（`/dev/sdb1`，ext4，label `usbdata`）上的原始包：

```text
/media/mnt-xm/usbdata/hagridv2_512/HaGRIDv2_dataset_512
```

该目录**只包含 JPG**（1,086,167 个文件），没有任何 JSON/JSONL/TXT/CSV 标注文件。
标注来自此前已存在于服务器上的官方 HaGRIDv2 标注缓存：

```text
data/cache/hagrid10-p1-05/annotations/{train,val,test}/{class}.json
```

标注与图片的对应关系在复制前做了完整核验，而不是抽样：

| 类 | 图片数 | train | val | test |
|---|---:|---:|---:|---:|
| one | 31,872 | 23,872 | 3,000 | 5,000 |
| peace | 31,801 | 23,801 | 3,000 | 5,000 |
| three | 30,721 | 22,721 | 3,000 | 5,000 |
| four | 31,436 | 23,436 | 3,000 | 5,000 |
| palm | 31,710 | 23,710 | 3,000 | 5,000 |
| like | 31,244 | 23,244 | 3,000 | 5,000 |
| ok | 31,153 | 23,153 | 3,000 | 5,000 |
| no_gesture | 2,164 | 1,464 | 200 | 500 |
| call | 28,061 | 20,061 | 3,000 | 5,000 |
| dislike | 31,624 | 23,624 | 3,000 | 5,000 |

核验方法：对每一类取标注 JSON 的全部 key 集合与图片 basename 集合，双向比对
（缺失、多余、跨 split 重复均为 0），并检查标注记录含 `bboxes`。结论
`ANNOTATION_KEY_CHECK_PASS`。

## 2. 复制到训练目录

只复制十类，不使用 `rsync` 逐文件（在 USB 上实测约 8 MB/s）。改用 `tar` 流式复制：

```bash
tar -C "$SRC" -cf - one peace three four palm like ok no_gesture call dislike \
  | tar -C "$DEST" -xpf -
```

目标：

```text
data/manual/hagrid_dataset/{one,peace,three,four,palm,like,ok,no_gesture,call,dislike}
data/manual/hagrid_annotations/{train,val,test}/{class}.json
```

复制完成后逐类比对源/目标文件数与字节数，全部相等：

```text
FAST_COPY_VERIFY_PASS total_dest_bytes=32996872829
```

十类合计 **32,996,872,829 字节（约 30.7 GiB）**，与移动硬盘源目录逐字节一致。

标注另行复制并做 SHA256 比对：30 个 JSON，共 1,028,020,206 字节，
`ANNOTATION_COPY_VERIFY_PASS`。

## 3. Pre-flight

对复制后的实际路径运行：

```bash
python3 -m scripts.p1_v2_local check \
  --images data/manual/hagrid_dataset \
  --annotations data/manual/hagrid_annotations
```

结果 `status = CHECK_OK`，`problems = []`，十类标注 key 抽样解析 20/20。

另外用 Pillow 对每类 20 张（共 200 张）做真实解码（`verify()` + `load()`）：
`200/200` 成功，短边全部为 `512`。这一步用于排除“扩展名是 JPG 但内容损坏”
或“复制了错误分辨率的包”。

## 4. 全量构建

原 builder 的 `--train/--val/--test-per-source` 是**上限**，最大 2000/2000，且
`MAX_PER_SOURCE = 2000` 会拒绝更全量的配置。为支持全量，新增了 `--full` 模式：

- 仅允许 `local-dir`；
- 不接受有界配额参数；
- 每个 source/split 取全部有效候选，而不是 2000 上限；
- 重复图片字节仍然剔除（同 split 内、跨 split 均不复制），并记入
  `dropped_cross_split_duplicate_bytes`；
- 有效但重复的候选被跳过时，不再错误地把它当成“样本不足”而整批失败。

命令：

```bash
python3 -m scripts.p1_v2_local build --full \
  --images data/manual/hagrid_dataset \
  --annotations data/manual/hagrid_annotations \
  --output data/processed/hagrid10-v2-512 \
  --license-ack data/reviews/hagrid-license-ack.json \
  --mapping-ack data/reviews/hagrid-mapping-ack.json \
  --max-output-gib 60
```

构建结果：

```text
status              = TEN_CLASS_CANDIDATE_BUILT_PENDING_HUMAN_QA
selection_mode      = full_available_valid_records
train / val / test  = 208,764 / 27,187 / 45,418
samples             = 281,369
invalid records     = train 19, val 1, test 7
dropped duplicates  = 34
skipped media       = 0
```

构建后对 281,369 张逐一做 SHA256 校验（图片 + YOLO label）：

```text
PRIVATE_DATASET_CHECKSUMS_PASS_HUMAN_QA_PENDING
```

即：文件完整性通过，人工视觉 QA 当时仍为 `PENDING`，`P1_pass = false`。

## 5. 数据集契约

十类输出顺序固定在 `dataset.yaml`：

```yaml
nc: 10
names:
  0: one
  1: two
  2: three
  3: four
  4: five
  5: like
  6: ok
  7: no_gesture
  8: call
  9: dislike
```

source 目录名与模型类别名的映射：

```text
peace -> two
palm  -> five
```

`dataset.yaml` 的 SHA256 由 builder 写入 `manifest.aggregate.json`，训练入口会重新
计算并比对；不一致直接拒绝训练。这样修改类别顺序或 split 路径无法绕过图片/label 哈希。

## 6. 人工 QA 与研究用途确认

对全量数据集按类确定性抽样 24 张/类，共 240 张，生成私有 QA 包：

```text
data/reviews/hagrid-v2-full-qa/
  index.html            抽检页（本地 Web UI）
  review.json           决策表单
  packet.manifest.json  抽样清单与哈希绑定
  images/               240 张副本
```

审核人在本地浏览器逐项确认后保存，结果为：

```text
samples_reviewed = 240
samples_ok       = 240
ok_rate          = 1.0
per_class_ok     = 每类 24
```

`verify` 与 `apply` 均通过：

```text
human_visual_decode_and_legal_qa_status = PASS
P1_pass                                 = false   # 此时仅完成视觉部分
```

随后显式确认研究用途范围并写入接受表：

```text
scope                  = research_only_training
accept_research_only_terms = true
commercial_use         = false
```

执行 `p1_dataset_qa accept` 后：

```text
P1_pass                = true
legal_clearance_status = RESEARCH_ONLY_TRAINING_ONLY
```

**注意**：`P1_pass = true` 在这里的含义是“在该研究用途范围内可训练”，
不表示商业许可已澄清，也不表示数据集已完成逐张全量人工审阅。
QA 是 240 张分层抽样，不是 281,369 张逐张确认。

## 7. 训练启动与第一次失败

首次训练命令在容器内以非 root 用户运行，`HOME=/tmp`：

```text
/data/manual/p2-full-v2-train.log
FULL_TRAIN_EXIT=1
```

失败原因不是数据或门禁：

```text
PermissionError: [Errno 13] Permission denied: '/tmp/Ultralytics/Arial.ttf'
ERROR Error writing to /tmp/Ultralytics/persistent_cache.json
RuntimeError: Dataset '...dataset.yaml' error ... Download failure for Arial.ttf
```

即 Ultralytics 试图把字体和设置写到 `/tmp/Ultralytics`，而该路径归属 root。
修复方式是把容器 `HOME`、`YOLO_CONFIG_DIR`、`MPLCONFIGDIR` 指向仓库内可写目录：

```text
data/manual/container-home
data/manual/container-home/Ultralytics
data/manual/mplconfig
```

第一次失败没有产生任何有效 epoch，`runs/p2-full-v2-200/weights` 为空，已保留为失败证据。

## 8. 正式训练

重启后使用新的 run 目录，避免与失败 run 混淆：

```bash
docker run --rm --gpus all --shm-size=2g \
  --user "$(id -u):$(id -g)" \
  --env HOME="$W/data/manual/container-home" \
  --env YOLO_CONFIG_DIR="$W/data/manual/container-home/Ultralytics" \
  --env MPLCONFIGDIR="$W/data/manual/mplconfig" \
  --env YOLO_AUTOINSTALL=False \
  --env PYTHONUNBUFFERED=1 \
  --volume "$W:$W" --workdir "$W" \
  espdet-gesture:torch220-cu121 \
  python -u -m scripts.p2_train train \
    --data "$W/data/processed/hagrid10-v2-512/dataset.yaml" \
    --model-yaml "$W/runs/config/espdet_pico_10.yaml" \
    --run-dir "$W/runs/p2-full-v2-200-r2" \
    --epochs 200 --batch 8 --workers 4 --device 0 --optimizer SGD
```

环境（来自训练日志）：

```text
Ultralytics 8.4.0
Python 3.10.13
torch 2.2.0
CUDA:0  NVIDIA GeForce RTX 4080  15,943 MiB
optimizer: SGD(lr=0.01, momentum=0.9), 95 weight / 101 weight-decay / 147 bias groups
dataloader workers: 4
```

关键超参数：

```text
imgsz      = 224
batch      = 8
epochs     = 200
optimizer  = SGD
lr0        = 0.01
lrf        = 0.01
momentum   = 0.9
weight_decay = 0.0005
seed       = 20261022
deterministic = true
amp        = false
mosaic     = 1.0
close_mosaic = 0
fliplr     = 0.5
mixup / copy_paste / degrees / flipud = 0
```

`optimizer` 必须显式给定：`auto` 会生成不含 ESPDet 自定义模块的参数组
（日志中曾出现 “0 weight”），导致卷积权重不更新而 loss 曲线仍然看似正常。
训练入口现已把优化器限制在白名单内，`auto` 不在其中。

## 9. 当前状态（截至本文写入时）

```text
tmux session : hagrid-v2-full-train-r2   RUNNING
run dir      : runs/p2-full-v2-200-r2
progress     : epoch 34/200
throughput   : 约 46–48 it/s，单 epoch 约 11–12 分钟
GPU          : RTX 4080，显存占用约 0.41 GiB，利用率约 40%
artifacts    : weights/best.pt, weights/last.pt 持续更新
```

预计总时长约 38–42 小时（200 epochs × 约 11–12 分钟，加验证开销）。

`weights/last.pt` 在部分 epoch 后即可用于续训，但当前 `p2_train.py` **未开放
`--resume` 参数**，直接重跑命令会从头开始。若需要断点续训，应先增加显式
resume 入口并验证 optimizer/scheduler/epoch 状态恢复。

## 10. 复现用的最小命令集

```bash
# 1. pre-flight
python3 -m scripts.p1_v2_local check \
  --images data/manual/hagrid_dataset \
  --annotations data/manual/hagrid_annotations

# 2. 全量构建
python3 -m scripts.p1_v2_local build --full \
  --images data/manual/hagrid_dataset \
  --annotations data/manual/hagrid_annotations \
  --output data/processed/hagrid10-v2-512 \
  --license-ack data/reviews/hagrid-license-ack.json \
  --mapping-ack data/reviews/hagrid-mapping-ack.json \
  --max-output-gib 60

# 3. 完整性校验
python3 -m scripts.p1_dataset_builder verify \
  --output data/processed/hagrid10-v2-512

# 4. 生成 QA 包
python3 -m scripts.p1_dataset_qa prepare \
  --dataset data/processed/hagrid10-v2-512 \
  --output data/reviews/hagrid-v2-full-qa \
  --per-class 24

# 5. 审核后在本地 Web UI 保存，然后验证并应用
python3 -m scripts.p1_dataset_qa verify \
  --dataset data/processed/hagrid10-v2-512 \
  --workspace data/reviews/hagrid-v2-full-qa \
  --review data/reviews/hagrid-v2-full-qa/review.json
python3 -m scripts.p1_dataset_qa apply \
  --dataset data/processed/hagrid10-v2-512 \
  --ack data/reviews/hagrid-v2-full-qa/qa-ack.json

# 6. 研究用途确认后放行训练
python3 -m scripts.p1_dataset_qa accept \
  --dataset data/processed/hagrid10-v2-512 \
  --acceptance data/reviews/hagrid-v2-full-qa/p1-acceptance.json \
  --license-ack data/reviews/hagrid-license-ack.json

# 7. 训练（与第 8 节的 docker run 等价）
python3 -m scripts.p2_train train \
  --data data/processed/hagrid10-v2-512/dataset.yaml \
  --model-yaml runs/config/espdet_pico_10.yaml \
  --run-dir runs/p2-full-v2-200-r2 \
  --epochs 200 --batch 8 --workers 4 --device 0 --optimizer SGD
```

## 11. 尚未完成

- 200 epoch 全量训练完成
- 独立 test 评估（mAP50 / mAP50-95 / P / R / 逐类 AP）
- ONNX 导出与 PyTorch/ONNX parity
- ESP-PPQ W8A8 量化
- CoreS3 实机部署与延迟/误触发验证

以上任一项都不能因为“训练已启动”而视为完成。
