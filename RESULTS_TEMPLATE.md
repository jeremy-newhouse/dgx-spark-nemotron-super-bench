# Validation result

Manual fallback only. `bench_decode.py` auto-generates a `validation-report.md` with
all of this filled in (and can file the issue for you) — prefer that. Use this template
if you can't run the script or want to send a partial report. Partial reports welcome.

## Hardware
- Device: DGX Spark (GB10) / other:
- Power mode / clocks (if known):
- Ambient / thermal notes:

## Software
- vLLM image (with digest if possible):
- Model + revision:
- Exact launch command, or deviations from `launch_server.sh`:

## Decode-short result  (batch=1, ~128-token prompt, 512-token forced decode, greedy, ignore_eos)
- iters:
- median decode tok/s:
- p10 / p90:
- median TTFT (s):

## MTP acceptance  (from /metrics)
- overall accepted / draft this run:
- per-position (pos0 / pos1 / pos2), if exposed:
- if unsure, paste every line containing `spec_decode` from `<host>:<port>/metrics`:

```
(paste here)
```

## Real-workload (optional)
- workload (e.g. a reasoning eval, long generations):
- effective decode tok/s:

## Notes / how this differed from the kit recipe
-
