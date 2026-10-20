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
