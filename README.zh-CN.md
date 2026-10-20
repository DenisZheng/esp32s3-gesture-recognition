# ESP32-S3 自训练十类手势检测

本仓库独立管理 **ESPDet-Pico + HaGRIDv2 -> 自主训练 -> ONNX -> ESP-PPQ -> ESP-DL -> M5Stack CoreS3** 的研究 PoC。

**当前阶段：P0 BLOCKED**：本地代码与技术文档已准备，但 GitHub Private 远端仓库、GitHub Actions 实际运行与 main 保护尚未完成，不能算 GitHub 初始化成功；也没有训练、量化或板卡验收证据。

本项目固定采用**单一目标检测网络**同时预测手部 Bounding Box、置信度和下列十类：`one, two, three, four, five, like, ok, no_gesture, call, dislike`。`no_gesture` 是有手部检测框的非命令手势，`no_hand` 不是第十一类。

已核查：官方 ESPDetect 结构可配置 `nc=10`，但 ESP-Detection 的量化评估样例写死 `NC=1`；ESP-DL v3.3.13 NMS 默认不判断类别 ID。二者必须在 P3/P4 经独立多类/多手测试确认。

核心文档：[总体计划](docs/PLAN.md) · [架构](docs/ARCHITECTURE.md) · [CoreS3摄像头基线](docs/BOARD_BASELINE.md) · [兼容性](docs/UPSTREAM_COMPATIBILITY.md) · [数据与隐私规则](docs/DATA_POLICY.md) · [验收规则](docs/ACCEPTANCE.md) · [许可](docs/THIRD_PARTY.md) · [风险](docs/RISKS.md)。

**P0 本地验证（无需数据集）：**

```bash
python3 scripts/check_contract.py
python3 scripts/check_repo_policy.py
python3 -m unittest discover -s tests -v
```

参考：[ESP-Detection](https://github.com/espressif/esp-detection)、[ESP-DL](https://github.com/espressif/esp-dl)、[HaGRID](https://github.com/hukenovs/hagrid)、[ESP-PPQ](https://github.com/espressif/esp-ppq)。所有上游在 `upstreams.lock.json` 固定 commit，仅在受控环境按需克隆。
