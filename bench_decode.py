#!/usr/bin/env python3
"""
Single-stream decode benchmark + validation-report generator for
Nemotron-3-Super-120B NVFP4 on DGX Spark.

Standard-library only (no pip installs). Measures steady-state decode throughput
against a running vLLM OpenAI-compatible server, reports speculative-decode (MTP)
acceptance from /metrics, harvests the environment needed to validate the result,
writes a paste-ready `validation-report.md` (+ `.json`), and -- if the GitHub CLI
is installed and authenticated -- offers to submit it as an issue for you.

  python3 bench_decode.py --iters 30                                      # realistic prompt (~26-27 t/s)
  python3 bench_decode.py --iters 30 --microbench                         # lorem-ipsum kernel microbench (~33.6)
  python3 bench_decode.py --iters 5 --max-tokens 8192 --temperature 1.0   # add the sampling effect
  python3 bench_decode.py --launch-cmd-file my-launch.sh                  # embed your exact command
  python3 bench_decode.py --submit                                        # file the issue without prompting

Methodology:
  - one request at a time (batch=1), streaming /v1/completions
  - temperature=0 (greedy), ignore_eos=true, max_tokens=512  -> deterministic length
  - decode tok/s = (completion_tokens - 1) / (t_last_token - t_first_token)
    i.e. prefill / TTFT is excluded; this is the steady-state decode rate
  - reports median, p10, p90 over N iters
For comparable GPU/clock details in the report, run on the GPU host (so nvidia-smi works).
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

REPO = "jeremy-newhouse/dgx-spark-nemotron-super-bench"

# Two prompt styles (select with --microbench). At batch=1 the prompt length only
# changes prefill/TTFT (excluded from the metric); what matters is content predictability,
# which sets MTP acceptance and therefore the decode rate.
#
#   realistic (default): a real instruction. Real prose is only moderately predictable, so
#       MTP acceptance is ~0.66 -- this is the representative single-stream number
#       (~26-27 tok/s), close to what real workloads see. This is the headline figure.
#   lorem (--microbench): repetitive lorem-ipsum filler whose greedy continuation is highly
#       predictable, so MTP acceptance saturates (~0.98). A decode-KERNEL microbench that
#       isolates raw decode throughput from content (~33.6 tok/s); it overstates real use.
# Same vLLM, same recipe, same kernels for both -- only the content differs.

_LOREM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis "
    "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat "
    "duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore "
).split()


def lorem_prompt(approx_tokens=128):
    """Deterministic lorem-ipsum filler tokenizing to ~approx_tokens (~3.5 chars/token)."""
    target_chars = int(approx_tokens * 3.5)
    out, n = [], 0
    while n < target_chars:
        out.append(_LOREM_WORDS[n % len(_LOREM_WORDS)])
        n += len(out[-1]) + 1
    return " ".join(out)


REALISTIC_PROMPT = (
    "You are a careful assistant. Write a detailed, step-by-step explanation of how "
    "a modern transformer language model generates text one token at a time, covering "
    "tokenization, the forward pass, attention over the key-value cache, sampling, and "
    "why memory bandwidth dominates single-stream decode on edge accelerators. Be "
    "thorough and precise, and continue until you have fully covered every part."
)


def http_json(url, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_metrics(base):
    try:
        with http_json(base + "/metrics") as r:
            return r.read().decode("utf-8", "replace")
    except Exception as e:
        print("  (could not fetch /metrics: %s)" % e, file=sys.stderr)
        return ""


# A Prometheus exposition line:  name{label="v",...} value [timestamp]
_SAMPLE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE.+-]+)(?:\s+\S+)?$')


def sum_metric(text, core, label_contains=None):
    """Sum samples whose metric name is exactly `core` or `core`+'_total'.

    Matching is on the exact metric name (a leading 'vllm:' is ignored), NOT a
    substring. A substring match on 'spec_decode_num_accepted_tokens' would also
    catch the per-position series ('..._per_pos_total{position=...}') -- whose
    samples sum to the same aggregate and change every step -- roughly doubling
    the count, plus the prometheus '_created' timestamp series. `label_contains`
    further restricts to samples whose label set contains the given text
    (e.g. 'position="0"').
    """
    total = 0.0
    found = False
    for line in text.splitlines():
        m = _SAMPLE_RE.match(line.strip())
        if not m:
            continue
        name = m.group(1)
        if name.startswith("vllm:"):
            name = name[len("vllm:"):]
        if name != core and name != core + "_total":
            continue
        if label_contains and label_contains not in (m.group(2) or ""):
            continue
        try:
            total += float(m.group(3))
            found = True
        except ValueError:
            pass
    return total if found else None


def compute_acceptance(m_before, m_after):
    """Speculative-decode (MTP) acceptance for tokens drafted during this run.

    Differences before/after isolate this run from prior history. Returns a dict
    (rate, accepted, draft, num_drafts, accept_len, per_pos) or None if the
    counters aren't present. Per-position rate[i] = accepted_at_pos_i / num_drafts
    (matches vLLM's own SpecDecodingLogging); overall rate = accepted / drafted
    tokens; accept_len = mean tokens emitted per forward pass (1 + accepted/drafts).
    """
    def delta(core, label=None):
        b = sum_metric(m_before, core, label)
        a = sum_metric(m_after, core, label)
        return None if (b is None or a is None) else (a - b)

    accepted = delta("spec_decode_num_accepted_tokens")
    draft = delta("spec_decode_num_draft_tokens")
    drafts = delta("spec_decode_num_drafts")
    if accepted is None or draft is None or draft <= 0:
        return None
    out = {
        "rate": accepted / draft,
        "accepted": accepted,
        "draft": draft,
        "num_drafts": drafts,
        "accept_len": (1 + accepted / drafts) if drafts else None,
        "per_pos": [],
    }
    if drafts and drafts > 0:
        for i in range(64):  # stop at the first position with no counter (= k)
            d = delta("spec_decode_num_accepted_tokens_per_pos", 'position="%d"' % i)
            if d is None:
                break
            out["per_pos"].append(d / drafts)
    return out


def run_cmd(args, timeout=30):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def cmd_out(args, timeout=20):
    r = run_cmd(args, timeout)
    return r.stdout.strip() if (r is not None and r.returncode == 0) else None


def smi_query(fields):
    out = cmd_out(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                   "--format=csv,noheader,nounits"])
    if not out:
        return None
    vals = [v.strip() for v in out.splitlines()[0].split(",")]
    return dict(zip(fields, vals))


def smi_snapshot():
    return smi_query(["temperature.gpu", "power.draw", "clocks.sm", "clocks.mem",
                      "utilization.gpu"])


def server_version(base):
    try:
        with http_json(base + "/version") as r:
            return json.loads(r.read()).get("version")
    except Exception:
        return None


def meminfo_total_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        return None


def collect_env(base, model):
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "vllm_version": server_version(base),
        "client_platform": platform.platform(),
        "client_arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "mem_total_gib": meminfo_total_gib(),
        "gpu": smi_query(["name", "driver_version", "memory.total", "power.max_limit",
                          "clocks.max.sm", "clocks.max.memory", "compute_cap"]),
        "cuda_version": None,
    }
    smi = cmd_out(["nvidia-smi"])
    if smi:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", smi)
        if m:
            env["cuda_version"] = m.group(1)
    return env


def one_iter(base, model, prompt, max_tokens, temperature=0.0):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first = None
    t_last = t0
    completion_tokens = None
    with http_json(base + "/v1/completions", payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            choices = obj.get("choices") or []
            if choices and choices[0].get("text"):
                if t_first is None:
                    t_first = now
                t_last = now
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens"):
                completion_tokens = usage["completion_tokens"]
    if t_first is None:
        raise RuntimeError("no tokens streamed")
    if not completion_tokens:
        completion_tokens = max_tokens  # include_usage unsupported; assume forced length
    decode_time = max(t_last - t_first, 1e-9)
    return (completion_tokens - 1) / decode_time, t_first - t0, completion_tokens


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def build_report(env, run, results, accept, snap_before, snap_after, launch_cmd):
    g = env.get("gpu") or {}

    def gv(k):
        v = g.get(k)
        return v if v not in (None, "", "[N/A]") else "n/a"

    def snap(d):
        if not d:
            return "n/a"
        return ("temp %s C, power %s W, SM %s MHz, mem-clk %s MHz, util %s%%"
                % (d.get("temperature.gpu"), d.get("power.draw"), d.get("clocks.sm"),
                   d.get("clocks.mem"), d.get("utilization.gpu")))

    if accept:
        acc_line = "%.3f overall (%.0f accepted / %.0f drafted tokens)" % (
            accept["rate"], accept["accepted"], accept["draft"])
        if accept.get("accept_len"):
            acc_line += "; mean accepted length %.2f tok/forward-pass" % accept["accept_len"]
        if accept["per_pos"]:
            acc_line += "; per-position [%s]" % ", ".join(
                "pos%d %.3f" % (i, r) for i, r in enumerate(accept["per_pos"]))
    else:
        acc_line = ("not derivable from /metrics on this build -- "
                    "paste the spec_decode lines manually")

    report = {"environment": env, "run": run, "results": results,
              "mtp_acceptance": accept,
              "gpu_snapshot_before": snap_before, "gpu_snapshot_after": snap_after,
              "launch_command": launch_cmd}

    md = """# Nemotron-3-Super NVFP4 DGX Spark -- decode validation report

Generated by `bench_decode.py` at {ts}.

## Result
- prompt: {pstyle}, mode: {mode}, max_tokens={maxtok}, iters={iters} (warmup {warmup})
- **decode tok/s median: {median:.2f}**
- p10 / p90: {p10:.2f} / {p90:.2f}
- mean: {mean:.2f}  (n={n})
- TTFT median (s): {ttft:.3f}
- MTP acceptance (this run): {acc}

## Environment
- vLLM version: {vllm}
- model: {model}
- GPU: {gname}, driver {gdrv}, CUDA {cuda}, compute_cap {gcc}, VRAM {gmem} MiB
- GPU limits: power max {gpmax} W, SM clock max {gsmmax} MHz, mem clock max {gmcmax} MHz
- GPU during run (before -> after): {sb}  ->  {sa}
- host: {plat} ({arch}), {cpus} CPUs, {mem} GiB RAM, Python {py}

## Recipe
{launch}

## Notes
<<FILL IN (optional): anything unusual, thermal/power mode, how this differed from the kit>>

---
Machine-readable JSON is in `validation-report.json`.
""".format(
        ts=env.get("timestamp_utc"), pstyle=run.get("prompt_style", "n/a"),
        mode=run["mode"], maxtok=run["max_tokens"], iters=run["iters"], warmup=run["warmup"],
        median=results["decode_tps_median"], p10=results["decode_tps_p10"],
        p90=results["decode_tps_p90"], mean=results["decode_tps_mean"], n=results["n"],
        ttft=results["ttft_median"], acc=acc_line,
        vllm=env.get("vllm_version") or "n/a", model=env.get("model"),
        gname=gv("name"), gdrv=gv("driver_version"), cuda=env.get("cuda_version") or "n/a",
        gcc=gv("compute_cap"), gmem=gv("memory.total"), gpmax=gv("power.max_limit"),
        gsmmax=gv("clocks.max.sm"), gmcmax=gv("clocks.max.memory"),
        sb=snap(snap_before), sa=snap(snap_after),
        plat=env.get("client_platform"), arch=env.get("client_arch"),
        cpus=env.get("cpu_count"), mem=env.get("mem_total_gib") or "n/a",
        py=env.get("python"), launch=launch_cmd,
    )
    return md, report


def gh_ready():
    if cmd_out(["gh", "--version"]) is None:
        return False
    r = run_cmd(["gh", "auth", "status"])
    return r is not None and r.returncode == 0


def maybe_submit(args, md_path, env, results, has_placeholders):
    repo = args.repo
    form_url = "https://github.com/%s/issues/new?template=validation-report.yml" % repo
    g = env.get("gpu") or {}
    title = "[validation] %s - %.1f tok/s single-stream decode" % (
        (g.get("name") or "GPU").strip(), results["decode_tps_median"])

    if args.no_submit:
        print("\nTo report: open %s and paste %s." % (form_url, md_path))
        return

    if not gh_ready():
        print("\nGitHub CLI not installed/authenticated. To report your result, do one of:")
        print("  - open %s and paste the contents of %s" % (form_url, md_path))
        print("  - install gh (https://cli.github.com), run `gh auth login`, then re-run with --submit")
        print("  - or share %s wherever you found this kit" % md_path)
        return

    if has_placeholders and not args.submit:
        print("\nYour report still has <<FILL IN>> placeholders (launch command / notes).")
        print("Fill them in %s, then re-run with --submit (or paste it into %s)." % (md_path, form_url))
        return

    do = args.submit
    if not do:
        if sys.stdin.isatty():
            ans = input("\nGitHub CLI is authenticated. Submit this report as an issue to "
                        "%s now? [y/N] " % repo).strip().lower()
            do = ans in ("y", "yes")
        else:
            print("\nNon-interactive shell; not submitting. Re-run with --submit, or open %s." % form_url)
            return
    if not do:
        print("Not submitted. Open %s and paste %s whenever you're ready." % (form_url, md_path))
        return

    cmd = ["gh", "issue", "create", "--repo", repo, "--title", title, "--body-file", md_path]
    r = run_cmd(cmd + ["--label", "validation"], timeout=60)
    if r is not None and r.returncode != 0 and "label" in (r.stderr or "").lower():
        r = run_cmd(cmd, timeout=60)  # retry without the label
    if r is not None and r.returncode == 0:
        print("\nSubmitted: %s" % r.stdout.strip())
    else:
        err = (r.stderr.strip() if r is not None else "gh not runnable")
        print("\ngh issue create failed (%s). Submit manually at %s" % (err, form_url))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None, help="defaults to the server's first model")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy. Try 1.0 with a large --max-tokens (e.g. 8192) to see "
                         "acceptance fall under sampling")
    ap.add_argument("--microbench", action="store_true",
                    help="run the lorem-ipsum decode-short kernel microbench (repetitive "
                         "content -> MTP acceptance ~0.98, ~33.6 tok/s) instead of the default "
                         "realistic prompt (acceptance ~0.66, ~26-27 tok/s)")
    ap.add_argument("--report", default="validation-report",
                    help="output prefix; writes <prefix>.md and <prefix>.json")
    ap.add_argument("--no-report", action="store_true", help="skip writing the report files")
    ap.add_argument("--launch-cmd-file", default=None,
                    help="path to a file with your exact launch/docker command; embedded in the report")
    ap.add_argument("--repo", default=REPO, help="GitHub repo for issue submission")
    ap.add_argument("--submit", action="store_true",
                    help="file the GitHub issue without prompting (requires gh installed + authed)")
    ap.add_argument("--no-submit", action="store_true", help="never offer to submit; just print instructions")
    args = ap.parse_args()

    base = "http://%s:%d" % (args.host, args.port)

    model = args.model
    if not model:
        try:
            with http_json(base + "/v1/models") as r:
                model = json.loads(r.read())["data"][0]["id"]
        except Exception as e:
            print("Could not auto-detect model (%s); pass --model." % e, file=sys.stderr)
            sys.exit(1)

    prompt = lorem_prompt() if args.microbench else REALISTIC_PROMPT
    prompt_style = "lorem (decode-short microbench)" if args.microbench else "realistic"
    mode = "greedy" if args.temperature == 0 else ("temp=%.2f" % args.temperature)
    print("Server: %s   model: %s" % (base, model))
    print("Shape:  batch=1, ~128-token %s prompt, %d-token forced decode, %s\n"
          % ("lorem-ipsum" if args.microbench else "realistic", args.max_tokens, mode))

    env = collect_env(base, model)

    for i in range(args.warmup):
        one_iter(base, model, prompt, args.max_tokens, args.temperature)
        print("  warmup %d/%d done" % (i + 1, args.warmup))

    snap_before = smi_snapshot()
    m_before = fetch_metrics(base)

    tps_list, ttft_list = [], []
    for i in range(args.iters):
        tps, ttft, ntok = one_iter(base, model, prompt, args.max_tokens, args.temperature)
        tps_list.append(tps)
        ttft_list.append(ttft)
        print("  iter %2d/%d:  %.2f tok/s   TTFT %.3fs   (%d tok)"
              % (i + 1, args.iters, tps, ttft, ntok))

    m_after = fetch_metrics(base)
    snap_after = smi_snapshot()

    results = {
        "decode_tps_median": statistics.median(tps_list),
        "decode_tps_p10": pct(tps_list, 0.10),
        "decode_tps_p90": pct(tps_list, 0.90),
        "decode_tps_mean": statistics.mean(tps_list),
        "ttft_median": statistics.median(ttft_list),
        "n": len(tps_list),
    }
    run = {"mode": mode, "max_tokens": args.max_tokens, "iters": args.iters,
           "warmup": args.warmup, "temperature": args.temperature,
           "prompt_style": prompt_style}

    accept = compute_acceptance(m_before, m_after)

    print("\n==================== RESULT ====================")
    print("decode tok/s  median: %.2f" % results["decode_tps_median"])
    print("              p10/p90: %.2f / %.2f" % (results["decode_tps_p10"], results["decode_tps_p90"]))
    print("              mean:    %.2f  (n=%d)" % (results["decode_tps_mean"], results["n"]))
    print("TTFT (s)      median: %.3f" % results["ttft_median"])
    if accept:
        print("MTP acceptance (this run): %.3f overall  (%.0f accepted / %.0f drafted tokens)"
              % (accept["rate"], accept["accepted"], accept["draft"]))
        if accept.get("accept_len"):
            print("              mean accepted length: %.2f tokens per forward pass"
                  % accept["accept_len"])
        if accept["per_pos"]:
            print("              per-position accept:  "
                  + "  ".join("pos%d %.3f" % (i, r) for i, r in enumerate(accept["per_pos"])))
        if accept["rate"] > 1.0:
            print("  WARNING: overall acceptance > 1.0 -- /metrics parsing is likely off for")
            print("  this build (metric names may have changed). Paste the raw 'spec_decode'")
            print("  lines from /metrics into your report so it can be reconciled.")
        elif accept["rate"] < 0.45:
            print("  WARNING: acceptance is low. ~0.33 means only the base token is accepted")
            print("  -> the MoE/activation path is likely mis-wired (a known failure mode for")
            print("  this model), not a tuning result. Flag this in your report.")
    else:
        print("MTP acceptance: could not derive from /metrics; paste 'spec_decode' lines manually.")
    if not env.get("gpu"):
        print("NOTE: nvidia-smi not found -- run on the GPU host for GPU/driver/clock details.")
    print("================================================")

    if args.no_report:
        return

    launch_cmd = "<<FILL IN: paste your exact `vllm serve` / `docker run` command, " \
                 "or write 'used launch_server.sh unmodified'>>"
    if args.launch_cmd_file:
        try:
            with open(args.launch_cmd_file) as f:
                launch_cmd = "```\n" + f.read().strip() + "\n```"
        except Exception as e:
            print("  (could not read --launch-cmd-file: %s)" % e, file=sys.stderr)

    md, report = build_report(env, run, results, accept, snap_before, snap_after, launch_cmd)
    md_path = args.report + ".md"
    with open(md_path, "w") as f:
        f.write(md)
    with open(args.report + ".json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote %s and %s.json" % (md_path, args.report))

    maybe_submit(args, md_path, env, results, "<<FILL IN" in md)


if __name__ == "__main__":
    main()
