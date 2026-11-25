#!/usr/bin/env python3
"""Reseed-test top sweep runs: retrain each sweep_<ts>_<NNNN>.ini config with several seeds.

Usage:
  python3 tests/reseed_new_sweep_tops.py 511 601 840 598
  python3 tests/reseed_new_sweep_tops.py 511 601 --seeds 11 22 33 --min-ts 1795000000000
  python3 tests/reseed_new_sweep_tops.py 511 601 --dry-run   # show matched logs, don't train
"""
import argparse, os, re, subprocess, sys, time, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUFFER = os.path.join(ROOT, "puffer")
LOG_DIR = os.path.join(ROOT, "logs", "breakout")
OLD_MAX = 1784664224203  # default --min-ts: ignore sweep logs older than this
SEEDS = [11, 22, 33, 44, 55]
TARGETS = []  # [(run_idx, note)], filled from argv

HP_KEYS = [
    "vec.total_agents", "vec.num_buffers", "vec.num_threads",
    "env.frameskip", "policy.hidden_size", "policy.num_layers",
    "train.total_timesteps", "train.learning_rate", "train.gamma", "train.gae_lambda",
    "train.replay_ratio", "train.clip_coef", "train.vf_coef", "train.vf_clip_coef",
    "train.max_grad_norm", "train.ent_coef", "train.momentum", "train.minibatch_size",
    "train.horizon", "train.prio_alpha", "train.prio_beta0",
    "train.vtrace_rho_clip", "train.vtrace_c_clip",
]


def parse_ini(path):
    data, section = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[f"{section}.{k.strip()}"] = v.strip()
    return data


def find_run(idx):
    # Newest matching log wins if several sweeps after OLD_MAX reuse the index.
    best = None
    for name in os.listdir(LOG_DIR):
        m = re.match(r"sweep_(\d+)_(\d+)\.ini$", name)
        if m and int(m.group(1)) > OLD_MAX and int(m.group(2)) == idx:
            if best is None or int(m.group(1)) > best[0]:
                best = (int(m.group(1)), name)
    return os.path.join(LOG_DIR, best[1]) if best else None


def load_cfg(idx):
    path = find_run(idx)
    if not path:
        raise FileNotFoundError(idx)
    d = parse_ini(path)
    sc = [float(x) for x in d["metrics.env/score"].split(",")]
    up = [float(x) for x in d["metrics.uptime"].split(",")]
    t850 = next((u for u, s in zip(up, sc) if s > 850), None)
    hp = {}
    for k in HP_KEYS:
        if k not in d:
            continue
        v = d[k]
        # int-valued policy dims (sweep may log floats)
        if k in ("policy.num_layers", "policy.hidden_size", "vec.total_agents",
                 "train.horizon", "train.minibatch_size", "env.frameskip"):
            v = str(int(float(v.replace("_", ""))))
        if k == "train.total_timesteps":
            v = str(int(float(v.replace("_", ""))))
        hp[k] = v
    hp.setdefault("vec.num_buffers", "1")
    hp.setdefault("vec.num_threads", "8")
    hp.setdefault("train.vtrace_rho_clip", "1.0")
    hp.setdefault("train.vtrace_c_clip", "1.0")
    return {
        "idx": idx,
        "src": os.path.basename(path)[:-4],
        "async": int(float(d.get("base.async", "1"))),
        "sweep_score": max(sc),
        "sweep_final": sc[-1],
        "sweep_uptime": up[-1],
        "sweep_t850": t850,
        "hp": hp,
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def run_train(cfg, seed):
    run_id = f"newtop_{cfg['idx']:04d}_s{seed}"
    log_path = os.path.join(LOG_DIR, f"{run_id}.ini")
    # Env is compiled into ./puffer; overrides must be --section.key=value.
    cmd = [
        PUFFER, "train",
        f"--base.run_id={run_id}",
        f"--base.seed={seed}",
        f"--base.async={cfg['async']}",
        "--base.checkpoint_interval=100000",
        *[f"--{k}={v}" for k, v in cfg["hp"].items()],
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"    FAIL rc={proc.returncode}: {proc.stderr[-300:]}", flush=True)
        return None
    d = parse_ini(log_path)
    sc = [float(x) for x in d["metrics.env/score"].split(",")]
    return {"score": sc[-1], "max": max(sc), "wall": wall}


def main():
    global OLD_MAX, SEEDS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=int, nargs="+", help="sweep run indices (the _NNNN suffix)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--min-ts", type=int, default=OLD_MAX,
                    help="only consider sweep_<ts>_*.ini with ts greater than this")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the matched log and sweep result per run, then exit")
    args = ap.parse_args()
    OLD_MAX, SEEDS = args.min_ts, args.seeds
    TARGETS[:] = [(i, f"sweep run {i:04d}") for i in args.runs]

    os.chdir(ROOT)
    results = []
    for idx, note in TARGETS:
        cfg = load_cfg(idx)
        print(f"\n=== run {idx:04d} — {cfg['src']} ===", flush=True)
        print(f"  sweep: final score={cfg['sweep_final']:.1f} uptime={cfg['sweep_uptime']:.2f}s "
              f"t850={cfg['sweep_t850']}", flush=True)
        print(f"  H={cfg['hp'].get('policy.hidden_size')} L={cfg['hp'].get('policy.num_layers')} "
              f"hzn={cfg['hp']['train.horizon']} clip={cfg['hp']['train.clip_coef']} "
              f"lr={cfg['hp']['train.learning_rate']} ent={cfg['hp']['train.ent_coef']}", flush=True)
        if args.dry_run:
            continue
        scores = []
        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            r = run_train(cfg, seed)
            if r:
                scores.append(r["score"])
                print(f"    score={r['score']:.1f} wall={r['wall']:.1f}s", flush=True)
            else:
                print("    FAIL", flush=True)
        if len(scores) >= 2:
            results.append({
                "idx": idx, "note": note, "sweep_score": cfg["sweep_score"],
                "sweep_t850": cfg["sweep_t850"], "scores": scores,
                "mean": mean(scores), "std": std(scores),
                "min": min(scores), "max": max(scores),
                "hp": cfg["hp"],
            })
            print(f"  → mean={mean(scores):.1f} std={std(scores):.1f} "
                  f"range={min(scores):.1f}-{max(scores):.1f} "
                  f"all≥800={sum(s>=800 for s in scores)}/{len(scores)} "
                  f"all≥850={sum(s>=850 for s in scores)}/{len(scores)}", flush=True)

    if args.dry_run:
        return 0
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'run':>5} {'sweep_t':>8} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} "
          f"{'≥800':>5} {'≥850':>5}  note")
    for r in results:
        t850 = r['sweep_t850'] if r['sweep_t850'] is not None else float('nan')
        print(f"{r['idx']:5d} {t850:8.3f} {r['mean']:7.1f} {r['std']:7.1f} "
              f"{r['min']:7.1f} {r['max']:7.1f} "
              f"{sum(s>=800 for s in r['scores']):2d}/{len(r['scores'])} "
              f"{sum(s>=850 for s in r['scores']):2d}/{len(r['scores'])}  {r['note']}")
        print(f"       scores={[round(s,1) for s in r['scores']]}")

    out = os.path.join(LOG_DIR, "seed_sensitivity", "new_sweep_tops_reseed.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("idx,note,sweep_score,sweep_t850,mean,std,min,max,scores,"
                "hidden,layers,horizon,clip,lr,ent,grad_norm,agents,frameskip\n")
        for r in results:
            h = r["hp"]
            f.write(
                f"{r['idx']},{r['note']!r},{r['sweep_score']},{r['sweep_t850']},"
                f"{r['mean']},{r['std']},{r['min']},{r['max']},"
                f"\"{' '.join(f'{s:.2f}' for s in r['scores'])}\","
                f"{h.get('policy.hidden_size')},{h.get('policy.num_layers')},"
                f"{h.get('train.horizon')},{h.get('train.clip_coef')},"
                f"{h.get('train.learning_rate')},{h.get('train.ent_coef')},"
                f"{h.get('train.max_grad_norm')},{h.get('vec.total_agents')},"
                f"{h.get('env.frameskip')}\n"
            )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
