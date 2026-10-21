# HaGRIDv2 custom public-license review (source audit, not legal sign-off)

Date: 2026-10-20. Source: hukenovs/hagrid Hagrid_v2-1M (pinned upstream commit cade4dd4ba162832ef2b62b92d89d4d7a2dc2b75).

**Evidence:** the original English PDF was downloaded on a temporary GitHub Actions runner and extracted with pdftotext. The five-page document was inspected, without adding the PDF or source image data to the project tree.

| Evidence item | Exact recorded value |
|---|---|
| Official license | https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/license/en_us.pdf |
| Source Git blob SHA1 | bbe714b77de6ce80dd7e09a086151688f33b6aff |
| PDF SHA256 | 14f4845e9c8d3de875cbac4139491da368ab1040a7f565e02894477279134d22 |
| PDF size and pages | 157632 bytes; 5 pages |
| Extraction evidence | https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39388322259 |

## Crucial legal identity

The HaGRID README refers to a variant of Creative Commons Attribution-ShareAlike 4.0, but **page 1 footnote explicitly states this custom text is NOT a Creative Commons license**. The PDF describes itself as a custom public license with attribution and conditions reserved. Do not replace it with the stock CC BY-SA 4.0 text or use the CC BY-SA label as a complete legal summary.

## Findings by actual PDF clause

1. **Section 1**: Defines Licensed Material, Adapted Material, public availability and a customized set of license elements. It does not specifically define neural model weights as Adapted Material.
2. **Section 2(a)(1)**: Worldwide, royalty-free, non-exclusive, non-sublicensable grant for copying, making available and creating/providing adapted material, subject to the terms. No express research-only, non-commercial or ban-on-training restriction found in the five-page text.
3. **Section 2(a)(4)-(5)**: Authorizes necessary technical changes and imposes conditions on subsequent recipients. This does not alone answer whether distributing trained weights invokes the share-alike condition.
4. **Section 2(b)**: Publicity, portrayal, privacy and personal rights are **not fully granted**; patents and trademarks are not included. Some limited waiver/undertaking applies only to rights held by the Licensor. Do not infer that depicted subjects have granted all biometric/personality rights.
5. **Section 3(a)**: When sharing original or modified licensed content, retain required creator attribution, copyright/license/disclaimer notices where provided, link to license/source as practicable, and notices of changes.
6. **Section 3(b)**: Sharing Adapted Material entails licensing conditions compatible with the custom license's attribution/conditions-preserved elements and prohibits additional restrictions. Whether particular ML weights are covered must be assessed independently.
7. **Section 4**: Database extraction/reuse rights and obligations are also addressed, referring to database rights under Russian Federation civil law.
8. **Sections 5-8**: As-is/no-warranty terms; automatic termination on infringement, with potential reinstatement within 30 days after discovery if cured; other interpretive limits.

**Technical/research assessment:** use of a small licensed HaGRID image/annotation subset for internal evaluation appears consistent with the broad copying/adaptation grant **if all applicable terms and privacy rights are satisfied**. This is preliminary and is not a legal opinion or a signed acceptance of the license by the user/employer.

**Commercial/model release assessment:** **NOT CLEARED**. Neither closed-source commercial model rights nor share-alike effects on model weights nor privacy rights can be determined from the dataset PDF alone. Upstream ESP-Detection/Ultralytics AGPL obligations are separate and must also be reviewed.

## P1 stop gate remains

The PDF's identity/source/hash/text audit is DONE. Actual authorized use requires the owner/reviewer to sign a local research-scope acknowledgement, containing the *exact* PDF SHA256 above, reviewer, valid review date and terms acceptance under ignored data/reviews/hagrid-license-ack.json. The tool rejects any other PDF hash. Do not commit signed acknowledgements or individual metadata to this public repo.

The second gate for image conversion is ten-class **visual** mapping signoff, also held locally. Tools and public CI cannot sign on behalf of a human reviewer. Issue #2 therefore remains open for actual owner/legal clearance, though the missing source PDF is now resolved.

## Minimal network acquisition analysis

On 2026-10-20 the official annotation URL returned HTTP 200 with Content-Length **719041314 bytes**, Accept-Ranges: bytes. A 65536-byte tail range returned 206 and a ZIP central directory reporting **121 entries**, offset 719028936 and size 12356 bytes. Exactly **30 JSON entries** match the ten model-source classes across train, val and test, with combined compressed sizes **171848608 bytes**. No individual JSON content or person metadata was fetched for this audit.

The three no_gesture annotation JSONs alone total **1401844 compressed bytes**, making them an inexpensive local first step *after user research-license acceptance*.

Provenance: https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39388994357

See P1_DATA_TOOLING.md for the bounded HTTP Range reader and local-only data-validation commands.


## Actual next owner action

The original HaGRIDv2 custom English PDF has already been source-reviewed and its SHA256 pinned above; there is no outstanding technical document fetch. The remaining **human** research license signoff and potential commercial/privacy/model-weights legal review cannot be done by GitHub CI. After owner research authorization, start the [P1-06 five-group private train-only visual review](P1_TEN_CLASS_VISUAL_REVIEW.md) rather than trying to sign all ten class mappings without actual example images.
