# CoreS3 camera and memory baseline (public-source audit, NOT board tested)

Target is **M5Stack CoreS3** (not CoreS3-Lite, S3-EYE, or generic S3). M5Stack documents 16MB flash, **8MB PSRAM**, a **GC0308 (0.3MP) camera**, 2.0-inch LCD and peripheral power-control path.

Sources:

- [M5Stack CoreS3 official hardware/IO documentation](https://docs.m5stack.com/en/core/CoreS3)
- [M5Stack camera example: GC0308 and RGB565 QVGA](https://docs.m5stack.com/en/arduino/m5cores3/camera)
- [Espressif CoreS3 BSP](https://github.com/espressif/esp-bsp/tree/master/bsp/m5stack_core_s3) (public BSP docs mention `esp_video` ~2.0 and GC0308)
- [CoreS3 BSP header](https://github.com/espressif/esp-bsp/blob/master/bsp/m5stack_core_s3/include/bsp/m5stack_core_s3.h)

## First P4 board tasks

1. Determine a **specific pinned release/commit** of the Espressif CoreS3 BSP and GC0308 camera driver compatible with ESP-IDF v5.4.4; record it in `upstreams.lock.json` **after a build smoke test**. Current BSP master is NOT pinned as a verified dependency.
2. Camera init must handle GC0308 reset/power expander and shared I2C; compare source pin map to real CoreS3 revision. In M5Stack's example, SCCB SCL=GPIO11 and SDA=GPIO12, VSYNC=GPIO46, HREF=GPIO38, PCLK=GPIO45 and DVP data include GPIO39..42,15,16,47,48. Do not blindly transplant Arduino-specific code to ESP-IDF.
3. Start with QVGA **RGB565** and a fixed test frame. Capture raw bytes, identify pixel endianness relative to ESP-DL S3 preprocessor `DL_IMAGE_CAP_RGB565_BIG_ENDIAN`. Validate red/green/blue reference patches and orientation/mirroring before any model parity conclusion.
4. Reserve camera FB(s) in 8MB PSRAM. Measure PSRAM free/min through capture+model lifecycle; LCD and network should be disabled for the minimal parity test to isolate memory, then added incrementally. Reduce framebuffer count or acquire/copy strategy if inference starves camera.
5. Compare preprocessed 224x224 RGB input pixel-by-pixel to the host implementation of letterbox=114 and /255 scaling, then run self-trained `.espdl`; only subsequently start live capture and LCD overlay.

**Source verification does not establish a successful CoreS3 camera driver build or hardware capability at the configured clock.** Those are P4 test gates.
