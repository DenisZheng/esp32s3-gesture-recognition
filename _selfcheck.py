"""Static self-check for _prefetch_all.py before it is allowed to run all weekend.

Runs offline: no network, no dataset reads. Verifies the control flow that the
previous hand-written attempts got wrong.
"""
import ast
import re
import sys

PATH = "_prefetch_all.py"
src = open(PATH, encoding="utf-8").read()
tree = ast.parse(src)
failures = []


def check(ok, label):
    print(("PASS  " if ok else "FAIL  ") + label)
    if not ok:
        failures.append(label)


funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
check({"main", "build_request_list", "pending_by_source", "cache_counts", "fail"} <= funcs,
      "all helper functions defined at module level")

main_src = src[src.index("def main()"):]
for helper in ("build_request_list", "pending_by_source", "cache_counts"):
    check(helper in main_src, f"main() calls {helper}")

# Progress accounting must be derived from files present AFTER the pass, not
# from the pass return value, or a resumed cache under-reports progress.
check("for r in todo" in main_src and ".exists()" in main_src,
      "progress measured from cache state after each source")

check("STALL_LIMIT" in src and "stalls >=" in src, "stall guard stops a fruitless loop")
check("stalls = 0" in main_src and "stalls += 1" in main_src,
      "stall counter resets on progress")

codes = sorted({int(m) for m in re.findall(r"return (\d)\b", src)})
check(codes == [0, 1, 2, 3], f"distinct exit codes for each outcome: {codes}")

check("ByteBudget(8 * 1024 ** 3)" in src, "index-read budget is separate and small")
check("ByteBudget(200 * 1024 ** 3)" in src, "transfer budget is large enough for multi-day retries")

# The builder itself is the authority on quota; this tool must not re-implement
# the per-split window differently.
check("margin = max(10, counts[\"train\"] // 10)" in src,
      "window margin matches the builder formula")
check("want = {s: min(MAX_PER_SOURCE, counts[s] + margin) for s in SPLITS}" in src,
      "window is computed per split")

# Refuse to publish anything: this is a cache warmer only.
for forbidden in ("dataset.yaml", "manifest.aggregate", "write_bytes(", "shutil.copy"):
    check(forbidden not in src, f"cache warmer never writes dataset artifacts ({forbidden})")

# Cache location must be validated by resolved path, not string prefix.
check("relative_to((ROOT / \"data\").resolve())" in src,
      "cache path validated against resolved data/ root")

check("sorted(grouped, key=lambda s: -len(grouped[s]))" in src,
      "largest gap downloaded first")

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED")
    sys.exit(1)
print("ALL STATIC CHECKS PASSED")
