# ESP32-S3 自训练十类手势识别（单模型）

本仓库独立管理 **ESPDet-Pico + HaGRIDv2 -> 自主训练 -> ONNX -> ESP-PPQ -> ESP-DL -> M5Stack CoreS3** 的研究 PoC。

**当前阶段：P0 GitHub 基础设施已完成**：Public 仓库、GitHub Actions 和 Active 的 `Protect main` 规则已核实。许可 PDF 全文已审阅并锁定 SHA256；实际使用许可确认与十类视觉映射仍待人工签核。**P1 数据工具已通过合成单测，真实 HaGRID 数据处理暂受许可确认阻断。**没有训练、量化或真实板卡验收证据。

本项目的**应用目标是手势识别**，采用**一个 ESPDet-Pico 目标检测网络**同时预测手部 Bounding Box、置信度和下列十类（不是检测+分类两个模型）：`one, two, three, four, five, like, ok, no_gesture, call, dislike`。`no_gesture` 是有手部检测框的非命令手势，`no_hand` 不是第十一类。

**输入尺寸：224×224（基线）**。官方 ESPDet-Pico 手部检测使用 224×224，官方 MobileNetV2 对裁剪后的手部进行分类时才使用 128×128；本项目使用一张完整摄像头画面、一个模型完成定位及识别，故优先从 224×224 开始。详见[输入尺寸选型](docs/INPUT_RESOLUTION.md)。

**ESP-DL 基线：v3.3.14**（2026-10-19 最新正式 Release，固定 Git SHA），ESP-IDF 仍为 v5.4.4。

已核查：官方 ESPDetect 结构可配置 `nc=10`，但 ESP-Detection 的量化评估样例写死 `NC=1`；ESP-DL v3.3.14 NMS 默认不判断类别 ID。二者必须在 P3/P4 经独立多类/多手测试确认。

核心文档：[总体计划](docs/PLAN.md) · [架构](docs/ARCHITECTURE.md) · [CoreS3摄像头基线](docs/BOARD_BASELINE.md) · [输入尺寸选型](docs/INPUT_RESOLUTION.md) · [兼容性](docs/UPSTREAM_COMPATIBILITY.md) · [数据与隐私规则](docs/DATA_POLICY.md) · [验收规则](docs/ACCEPTANCE.md) · [许可](docs/THIRD_PARTY.md) · [风险](docs/RISKS.md)。

**P0 本地验证（无需数据集）：**

```bash
python3 scripts/check_contract.py
python3 scripts/check_repo_policy.py
python3 -m unittest discover -s tests -v
```

参考：[ESP-Detection](https://github.com/espressif/esp-detection)、[ESP-DL](https://github.com/espressif/esp-dl)、[HaGRID](https://github.com/hukenovs/hagrid)、[ESP-PPQ](https://github.com/espressif/esp-ppq)。所有上游在 `upstreams.lock.json` 固定 commit，仅在受控环境按需克隆。


P1 数据审计与微样本转换用法：[docs/P1_DATA_TOOLING.md](docs/P1_DATA_TOOLING.md)。需真实许可批准及全部十类人工映射签核，工具不会自动下载或使用 HaGRID 数据。


HaGRIDv2 使用自行改写的许可，**不是标准 CC BY-SA 4.0**。详见 [完整许可审查记录](docs/HAGRID_LICENSE_AUDIT.md)。


## P1 数据筛选补充

HaGRIDv2 官方共 **33 种手势 + 1 个 no_gesture，合计 34 类原始数据**；本项目的 ESPDet-Pico 始终只输出 **10 类**。其它 24 类不能全部作为负样本。当前划分为 7 类需单独审核的单手非指令候选、10 类形态易混淆而暂缓的类别、7 类双手组合暂缓类别。

特别注意 ESP-DL 的数字 `two` 是分开的 V 形，而 HaGRID `two_up` 双指更靠拢、`peace` 更像 V 形。正式批准之前，不允许把 `peace` 当作 `no_gesture` 训练。参见 [P1 难负样本政策](docs/HARD_NEGATIVES.md)。


**P1-03 本地视觉验收工具：** 已增加 HTML 对照页，展示 `two_up`、`peace` 等手势的逐手检测框，并生成图片哈希绑定的待签核记录；只使用训练划分、不读取测试集、无需向公共 GitHub 上传真实图片。说明见 [P1_VISUAL_REVIEW.md](docs/P1_VISUAL_REVIEW.md)。


**P1-04 数据下载优化：** 已实测 HaGRIDv2 的 `two_up.zip` 约 42.73 GB，索引含 30,688 张 JPEG。新增 ZIP64 按字节提取工具，可在实际授权后仅选取训练集少量图片供本地十类映射验收，无需下载整个类别压缩包。公开 GitHub CI 仅检查元数据，未下载人物图片。参见 [P1_IMAGE_ZIP64.md](docs/P1_IMAGE_ZIP64.md)。


**P1-04 四类对照源均已实测：** `two_up`、`peace`、`peace_inverted`、`two_up_inverted` 的官方 ZIP64 索引均通过 GitHub Actions 真实读取，归档精确大小及中央索引 SHA256 已固定，数据源更改会拒绝下载。真实人物图片依然没有下载，许可签核和人工图像验收仍未完成。 [验证记录](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39420065430)。


**新增 P1 一键本地验收入口：** `python3 -m scripts.p1_review_workflow plan` 可离线查看所需资料；授权人员完成许可签核后，`prepare` 才能在本地一次准备四类训练样本、JPEG 和带检测框的 HTML 报告；`verify` 校验人工记录和图片哈希。失败会回滚，公共 CI 严禁下载真实标注/图片。详细操作见 [P1_ONE_CLICK_REVIEW.md](docs/P1_ONE_CLICK_REVIEW.md)。本功能暂不代表十类映射/P1 已通过。


## P2-00 本地训练环境准备（不代表已开始 P2 正式训练）

自训练 ESPDet-Pico 必须在电脑或云 GPU 上运行，ESP32-S3 只负责训练后量化模型推理。推荐 Windows + WSL2 Ubuntu + NVIDIA CUDA 或 Linux GPU。新增 [P2 本地训练环境与自检指南](docs/P2_HOST_SETUP.md)，提供 `python3 -m scripts.p2_host_doctor plan`、GPU 依赖自检和 `p2_graph_smoke` 单模型十类随机输入反向传播测试。已发现官方上游对 `ultralytics>=8.3.112` 的下限存在兼容性缺口，v8.4.0 的代码支持 ESPDet 所需 `reg_max` 参数，但尚需真正运行验证。P1 许可、真实手势语义及数据验收通过前不启动模型训练。


**2026-10-21 CPU 实测通过：** 已在真实 Linux CPU 的 PyTorch 2.2.0 + Ultralytics 8.4.0 环境，使用固定版本 ESPDet-Pico 构造十类网络，完成 224×224 随机输入前向及反向传播，自定义分类头 42 个参数张量产生有效梯度。[GitHub Actions 实测](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39523052733)。**尚未测试你的 NVIDIA CUDA 机器、真实 YOLO 检测损失或 HaGRID 正式训练。**


### P2-00 现有 Linux GTX 1060 Docker 训练机

根据已有 SCRFD 自训练手册，可以直接复用 **Linux + 32GB 内存 + GTX 1060 6GB + Docker/NVIDIA Container Toolkit**，不必重新安装 WSL2，也不能覆盖 SCRFD 旧的 PyTorch 1.8/CUDA 11.1/MMCV 镜像。新增 [GTX 1060 专用 Docker 环境和验收指南](docs/P2_GTX1060_DOCKER.md)，新镜像使用 PyTorch 2.2.0 + CUDA 12.1 + Ultralytics 8.4.0。依次通过 `build → gpu → source → doctor → smoke` 验证，**当前 GitHub 只做离线合成检查，真实 1060 CUDA 仍待你在 Linux 主机上执行**；许可签核和真实数据验收之前不启动正式训练。


### P1-05 双模式训练集准备

已新增 [自动按需下载／手动 ZIP／本地目录统一构建指南](docs/P1_DUAL_SOURCE_DATASET.md)。`python3 -m scripts.p1_dataset_builder plan` 离线预览；经数据授权和十类真实视觉映射签核后，可用 `build --mode auto` 断点续取官方 ZIP 的目标图片，也可用 `--mode local-zip` 直接读取手动下载的每类原始 ZIP，或 `--mode local-dir` 读取已解压目录。输出标准 10 类 YOLO 检测训练集，按 `user_id` 校验原始 train/val/test 隔离，保留逐手检测框，哈希校验；**不能代替人工图片 QA、HaGRID 许可和正式训练验收**。`no_gesture` 使用官方另一地址的小 ZIP32 包，已专门兼容。
