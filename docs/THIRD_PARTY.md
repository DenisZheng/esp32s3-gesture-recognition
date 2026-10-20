# Third-party authorization and commercial-risk inventory

**Research PoC ONLY, commercial clearance is currently BLOCKED.** This is a source register, not a legal opinion or permission to ship. Private-repo status does not avoid AGPL or dataset license obligations. A separately reviewed license / provenance matrix is required before distributing firmware, weights, training code, images or a commercial product.

| Component | Source | Observed license / state | Research usage & commercial concern |
|---|---|---|---|
| ESP-Detection + copied/modified ESPDetect source if any | [espressif/esp-detection](https://github.com/espressif/esp-detection/blob/main/LICENSE) | AGPL-3.0 | Training and export code are AGPL. Reusing or modifying protected code and distributing covered software can trigger source-availability/notice obligations; network-service use can also matter. **Do not copy whole project or assert Private means exempt.** |
| Ultralytics | [Ultralytics license](https://www.ultralytics.com/license) | AGPL-3.0 or separate enterprise/commercial arrangements | Determine exactly which library/version is installed and whether distributed model, runtime, or combined work creates obligations. A commercial license may be needed for closed-source product strategy; seek legal analysis. |
| ESP-DL v3.3.14 | [espressif/esp-dl](https://github.com/espressif/esp-dl/blob/v3.3.14/LICENSE) | MIT at repo and component | Include attribution/notice for redistributed code; board-specific dependencies may have their own licenses. |
| ESP-PPQ | [espressif/esp-ppq](https://github.com/espressif/esp-ppq/blob/master/LICENSE) | Apache-2.0 at repo | Notices, modification marking and license compliance as applicable; separate training tool from runtime artifact rights. |
| HaGRIDv2 image dataset and annotation metadata | [hukenovs/hagrid](https://github.com/hukenovs/hagrid) and [`license/en_us.pdf`](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf) | README states **variant of CC BY-SA 4.0**; **custom PDF terms not yet fully audited** | Need exact rights (research, redistribution, commercial use, adaptation/sharealike, attribution, model training and weights) before acquiring and using any restricted data. **Do not assume standard CC BY-SA or automatic model commercial rights.** |
| HaGRID pretrained models | Dataset repository model links | Separate weight attribution/terms unclear; NOT used in scratch baseline | Baseline starts from random weights; only allow a later explicitly licensed comparison experiment with full provenance. |
| CoreS3 BSP, camera sensor drivers, ESP-IDF | M5Stack and Espressif sources | Per-component license mix | Freeze board component, sensor/pin configs; review each included SDK/BSP/third-party driver before distribution. |

## Policy

1. Do not upload third-party HaGRID image bytes, raw individual annotations/subject IDs, checkpoint files (`.pt`, `.onnx`, `.espdl`), or personal test media to GitHub.
2. Do not imply that self-trained weights are automatically independent of AGPL or dataset terms. Derivative model licensing and output rights can be context-dependent; request qualified counsel for commercial use.
3. Before P1 licensed subset download, **read the exact HaGRID custom `en_us.pdf` and capture any additional restrictions in a signed internal record**. README alone is not enough. If download rights are not established, P1 is BLOCKED.
4. When any code is adapted from AGPL/MIT/Apache upstream, preserve source provenance and file-level notices, comply with the actual license and track changes. A Git submodule/pin does not remove compliance duties.
5. Before any production distribution, perform an SBOM review with transitive libraries, weight provenance, data release permissions, privacy and notices; seek license-specific commercial clearance.

`upstreams.lock.json` records revision and observed top-level license names. It is **not** an SPDX-complete dependency tree and is not clearance.
