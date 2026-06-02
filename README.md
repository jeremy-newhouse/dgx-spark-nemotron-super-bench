# Nemotron-3-Super-120B NVFP4 on DGX Spark — decode benchmark & validation kit

A minimal, reproducible kit to measure **single-stream decode throughput** of
[`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
on a single **NVIDIA DGX Spark** (GB10, sm_121a) under **vLLM** with the model's
built-in **MTP** speculative-decoding head.

I'm publishing this so the numbers below are **independently reproducible**: the exact
launch recipe plus a dependency-light benchmark, so anyone with a Spark can run it and
get the same figures (or find where their rig differs). If you do run it, the script can
file a GitHub issue with your results.

## What I measure

Single-stream (batch=1) decode throughput, **decode-only** (TTFT excluded), median of N
iters, with the model's MTP speculative-decoding head. The decode rate is coupled to MTP
acceptance, which depends on how predictable the generated content is — so the prompt matters:

| Run | My result | What it is |
|---|---|---|
| Realistic single-stream (default) | **~26–27 tok/s** | real-English prompt, 512-token greedy decode. MTP acceptance ~0.66 (per-position ≈ 0.84 / 0.65 / 0.49). **This is what the kit reproduces by default** and the number I'd point to. |
| Real-workload | ~27–32 tok/s | long reasoning generations (AIME-style); acceptance ~0.72 (per-position ≈ 0.87 / 0.71 / 0.58 at k=3). The default run above is a close, lower-effort proxy for this. |
| Decode-short microbench (`--microbench`) | ~33.6 tok/s (v0.22.0); 34.52 (earlier cu130-nightly) | lorem-ipsum prompt → repetitive content → acceptance ~0.98. A best-case **kernel-path** number that isolates decode throughput from content; it *overstates* real use. |

Same vLLM, same recipe, same kernels across all three — only the content (and therefore
MTP acceptance) differs.

Notes:
- The headline ~26–27 tok/s is a realistic-content single-stream number; the ~27–32 tok/s
  real-workload figure (AIME-style reasoning) is close, since both reflect real content
  difficulty (~0.66–0.72 acceptance).
- The lorem-ipsum `--microbench` (~33.6) is a kernel-path best case: it maxes MTP
  acceptance to isolate raw decode throughput, so it *overstates* real-workload yield.
- These aren't directly comparable to some published single-Spark figures: vLLM's DGX
  Spark blog (~22.7–23.7 tok/s) measured with **speculative decoding disabled**, and the
  community "airawatraj" recipe (23.45 tok/s) used a different setup. I'm not making a
  comparative claim — the methodologies differ.
- Numbers are from one rig and one image; clocks and thermals vary between Sparks.

## Requirements
- 1× NVIDIA DGX Spark (GB10, sm_121a; 128 GB unified LPDDR5x)
- Docker + the NVIDIA Container Toolkit
- The model `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (~74 GB; HF login may be required)
- vLLM. I use the stable `vllm/vllm-openai:v0.22.0` image. NVIDIA's NGC
  `nvcr.io/nvidia/vllm:26.05` also runs the model but ships an older vLLM (0.20.1).
- Python 3.9+ on the client side (the benchmark is **standard-library only**)
- Optional: the [GitHub CLI](https://cli.github.com) (`gh`), if you want the script to
  file your validation report automatically

## Quickstart
1. Edit `launch_server.sh` — set `MODEL` (HF id or local path) and review the env/flags.
2. **Drop the Linux page cache first** — otherwise the loader can OOM this unified-memory
   box: `sudo sysctl vm.drop_caches=3`
3. Start the server: `./launch_server.sh`. A preflight checks free memory and **stops with
   instructions if it's too low** (e.g. if the cache wasn't dropped or other processes are
   holding RAM). Wait for `Application startup complete` (first load takes several minutes
   while weights JIT for sm_121a).
4. In another shell, run the benchmark: `python3 bench_decode.py --iters 30`
5. The script writes `validation-report.md` and offers to file it as a GitHub issue (if
   you have `gh`); otherwise it prints how to report manually. See **Reporting results**.

## The recipe (what produces the number)

vLLM `v0.22.0`, single DGX Spark, batch=1. Environment:

```
VLLM_NVFP4_GEMM_BACKEND=marlin     # dense GEMM backend = Marlin NVFP4
VLLM_USE_FLASHINFER_MOE_FP4=0      # FlashInfer FP4 MoE is broken on Spark; force off
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

`vllm serve` flags:

```
--quantization modelopt_fp4
--kv-cache-dtype fp8
--mamba_ssm_cache_dtype float32
--max-model-len 32768              # decode rate is insensitive to this; NVIDIA's guide uses 1048576
--gpu-memory-utilization 0.90
--max-num-seqs 1                   # single-stream; NVIDIA recommends 4 for Spark stability
--speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
```

**Important:** do *not* force a Marlin MoE override. Leaving the MoE backend to vLLM's
auto-pick (native CUTLASS NVFP4) is the fast, stable path on v0.22.0; forcing Marlin
MoE triggers a large weight-repack that can OOM the unified-memory box.

### Memory & OOM safety
The Spark shares one 128 GB pool between GPU and host (unified memory), so an
over-committed launch can OOM the *host*, not just the GPU. The numbers above were
measured at `--gpu-memory-utilization 0.90`; that stays safe here because the recipe:
- uses `--max-model-len 32768` (not the 1M in NVIDIA's guide), so KV reservation is small;
- leaves the MoE backend on CUTLASS auto-pick (forcing a Marlin MoE repack spikes ~37 GB);
- checks free memory in the launch preflight and stops if it's too low (drop the page
  cache yourself first: `sudo sysctl vm.drop_caches=3`), and runs the container with
  `--oom-score-adj 1000` so vLLM is the first process killed under pressure.

If you still hit memory pressure, lower it: `GPU_MEM_UTIL=0.80 ./launch_server.sh`. At
batch=1 this does **not** change the decode rate — it only shrinks the KV cache. The
preflight's free-memory floor is tunable too (`MIN_AVAIL_GIB`, default 90) and can be
bypassed with `FORCE=1` (not recommended on a remote Spark).

## How the benchmark works
`bench_decode.py` sends one streaming `/v1/completions` request at a time
(`temperature=0`, `ignore_eos=true`, `max_tokens=512`, `stream_options.include_usage=true`).
By default it uses a realistic English prompt; pass `--microbench` for the lorem-ipsum
decode-short kernel microbench.
- It records time-to-first-token (TTFT) and total time, then reports
  **decode tok/s = (completion_tokens − 1) / (t_last − t_first_token)** — the
  steady-state decode rate, with prefill/TTFT excluded.
- It scrapes `/metrics` before and after the run and reports speculative-decode
  acceptance: overall, per-position, and mean accepted length (tokens per forward pass),
  so you can confirm MTP is firing and see how content affects it.
- It reports median, p10, p90 across N iterations, and writes a report (below).

**MTP health tripwire:** a healthy default run shows per-position acceptance well above the
floor (realistic ≈ 0.84 / 0.65 / 0.49; the lorem-ipsum `--microbench` saturates ≈ 0.95+).
If acceptance is pinned near **0.33** (only the base token accepted), the MoE/activation
path is mis-wired — a known failure mode for this model, not a tuning result.

**The microbench, and where the numbers come from.** The default realistic run (~26–27
tok/s, acceptance ~0.66) closely tracks the ~27–32 tok/s I see on real AIME-style
reasoning (~0.72) — most of the difference from the lorem-ipsum microbench is *content*
predictability, not temperature. To see the kernel-path best case:

```
python3 bench_decode.py --iters 30 --microbench
```

(~33.6 tok/s, acceptance ~0.98). Adding `--temperature 1.0 --max-tokens 8192` to either
run layers on the smaller sampling effect.

## Reporting results
Running the benchmark writes a paste-ready `validation-report.md` (plus a
machine-readable `validation-report.json`) that captures your environment — GPU,
driver, CUDA, clocks/power before and after the run (to catch thermal throttling),
vLLM version, host — alongside the decode results and MTP acceptance.

Then pick whichever is easiest:
1. **Automatic (preferred).** If you have the GitHub CLI installed and authenticated
   (`gh auth login`), the script offers to open the issue for you — or run it with
   `--submit` to file it without prompting.
2. **Manual issue.** Open the **Validation report** issue form and paste
   `validation-report.md`:
   <https://github.com/jeremy-newhouse/dgx-spark-nemotron-super-bench/issues/new?template=validation-report.yml>
3. **Pull request (optional).** Add your `validation-report.json` to `submissions/` via
   PR for an aggregated dataset (see [`submissions/README.md`](submissions/README.md)).

Please leave out host names or other personal identifiers in what you submit. Fill in
the launch-command and notes placeholders in the report before sending.

## License
MIT — see [LICENSE](LICENSE).
