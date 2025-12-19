import json
from .settings import Config
from .benchmark import run_baseline, run_speculative

def run_experiment(models, dataset, wer_metric, top_p_values, draft_k_values, config):
    results, baseline_cache, refs, count = [], {}, None, 0
    total = len(top_p_values) * len(draft_k_values)
    
    for top_p in top_p_values:
        if top_p not in baseline_cache:
            cfg = Config(top_p=top_p, temperature=config.temperature, max_new_tokens=config.max_new_tokens, language=config.language, task=config.task)
            print(f"Running baseline (top_p={top_p})...")
            base_res = run_baseline(dataset, models, cfg)
            baseline_cache[top_p] = {"avg_time": base_res["avg_time"], "wer": wer_metric.compute(predictions=base_res["predictions"], references=base_res["references"])}
            refs = base_res["references"]
        
        base = baseline_cache[top_p]
        for draft_k in draft_k_values:
            count += 1
            print(f"[{count}/{total}] top_p={top_p}, draft_k={draft_k}")
            cfg = Config(top_p=top_p, draft_k=draft_k, temperature=config.temperature, max_new_tokens=config.max_new_tokens, language=config.language, task=config.task)
            
            spec_res = run_speculative(dataset, models, cfg)
            wer = wer_metric.compute(predictions=spec_res["predictions"], references=refs)
            speedup = base["avg_time"] / spec_res["avg_time"]
            
            results.append({
                "top_p": top_p, "draft_k": draft_k,
                "baseline_avg_time": round(base["avg_time"], 4), "speculative_avg_time": round(spec_res["avg_time"], 4),
                "speedup": round(speedup, 2), "baseline_wer": round(base["wer"], 4), "speculative_wer": round(wer, 4)
            })
            print(f"    Speedup: {speedup:.2f}x | WER: {wer:.4f}")
    
    results_sorted = sorted(results, key=lambda x: x["speedup"], reverse=True)
    return {
        "config": {"top_p_values": top_p_values, "draft_k_values": draft_k_values, "dataset_size": len(dataset), "max_new_tokens": config.max_new_tokens, "temperature": config.temperature},
        "results": results_sorted,
        "best": {
            "speedup": max(results, key=lambda x: x["speedup"]),
            "wer": min(results, key=lambda x: x["speculative_wer"]),
            "closest_2x": min(results, key=lambda x: abs(x["speedup"] - 2.0))
        }
    }

def save_results(results, filepath):
    with open(filepath, "w") as f: json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")

def print_results(results):
    print("\n" + "=" * 80 + "\nEXPERIMENT RESULTS\n" + "=" * 80)
    print(f"\n{'top_p':>8} | {'draft_k':>8} | {'Base(s)':>8} | {'Spec(s)':>8} | {'Speedup':>8} | {'WER':>8}\n" + "-" * 70)
    for r in results["results"]:
        print(f"{r['top_p']:>8.2f} | {r['draft_k']:>8d} | {r['baseline_avg_time']:>8.3f} | {r['speculative_avg_time']:>8.3f} | {r['speedup']:>7.2f}x | {r['speculative_wer']:>8.4f}")
    
    print("\n" + "=" * 80 + "\nBEST CONFIGURATIONS\n" + "=" * 80)
    b = results["best"]
    print(f"\nBest Speedup:  top_p={b['speedup']['top_p']}, draft_k={b['speedup']['draft_k']} -> {b['speedup']['speedup']}x (WER: {b['speedup']['speculative_wer']})")
    print(f"Best WER:      top_p={b['wer']['top_p']}, draft_k={b['wer']['draft_k']} -> {b['wer']['speedup']}x (WER: {b['wer']['speculative_wer']})")
    print(f"Closest to 2x: top_p={b['closest_2x']['top_p']}, draft_k={b['closest_2x']['draft_k']} -> {b['closest_2x']['speedup']}x")
