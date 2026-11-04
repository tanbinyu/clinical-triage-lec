import json
import os
import argparse

from pipeline.config import NUM_DISEASES, DISEASE_NAMES


def evaluate_lec_result(result, alpha):
    splits = result["splits"]

    valid_splits = [s for s in splits if s["t_hat"] is not None]
    infeasible_count = len(splits) - len(valid_splits)

    fdr_list = []
    exceed_count = 0

    for s in valid_splits:
        selected = s["selected_num"]
        errors = s["error_num"]
        if selected == 0:
            fdr = 0.0
        else:
            fdr = errors / selected
        fdr_list.append(fdr)
        if fdr > alpha:
            exceed_count += 1

    if not fdr_list:
        return {
            "alpha": alpha,
            "total_epochs": len(splits),
            "valid_epochs": 0,
            "infeasible_epochs": infeasible_count,
            "mean_fdp": None,
            "std_fdp": None,
            "max_fdp": None,
            "mfdr": None,
            "exceed_count": 0,
            "exceed_rate": 0.0,
            "controlled": None,
        }

    mean_fdp = sum(fdr_list) / len(fdr_list)
    std_fdp = (sum((x - mean_fdp) ** 2 for x in fdr_list) / len(fdr_list)) ** 0.5
    max_fdp = max(fdr_list)

    total_errors = sum(s["error_num"] for s in valid_splits)
    total_selected = sum(s["selected_num"] for s in valid_splits)
    mfdr = total_errors / total_selected if total_selected > 0 else 0.0

    return {
        "alpha": alpha,
        "total_epochs": len(splits),
        "valid_epochs": len(valid_splits),
        "infeasible_epochs": infeasible_count,
        "mean_fdp": round(mean_fdp, 6),
        "std_fdp": round(std_fdp, 6),
        "max_fdp": round(max_fdp, 6),
        "mfdr": round(mfdr, 6),
        "exceed_count": exceed_count,
        "exceed_rate": round(exceed_count / len(valid_splits), 4),
        "controlled": mean_fdp <= alpha,
    }


def print_eval(label, ev):
    controlled_str = "PASS" if ev["controlled"] else "FAIL"
    if ev["controlled"] is None:
        controlled_str = "N/A"
    print(f"  [{controlled_str}] {label}")
    print(f"    alpha={ev['alpha']}, mean_fdp={ev['mean_fdp']}, std_fdp={ev['std_fdp']}, max_fdp={ev['max_fdp']}")
    print(f"    valid={ev['valid_epochs']}/{ev['total_epochs']}, infeasible={ev['infeasible_epochs']}, exceed={ev['exceed_count']}({ev['exceed_rate']})")


def run(output_dir, beta, alpha_A, alpha_B):
    results = {}

    p1_path = os.path.join(output_dir, "step3", f"lec_phase_1_beta_{beta}.json")
    if os.path.exists(p1_path):
        with open(p1_path) as f:
            p1 = json.load(f)
        ev = evaluate_lec_result(p1, beta)
        results["phase_1"] = ev
        print("=" * 60)
        print("Phase 1 (Routing)")
        print("=" * 60)
        print_eval(f"lec_phase_1_beta_{beta}", ev)
    else:
        print(f"[WARNING] Not found: {p1_path}")

    print()
    print("=" * 60)
    print(f"Phase 2 - Branch A (AI routing, alpha_A={alpha_A})")
    print("=" * 60)
    results["branch_a"] = {}
    for k in range(NUM_DISEASES):
        path = os.path.join(output_dir, "step6", f"lec_branch_a_disease_{k}_beta_{beta}_alphaA_{alpha_A}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            ev = evaluate_lec_result(data, alpha_A)
            results["branch_a"][k] = ev
            print_eval(f"disease_{k} ({DISEASE_NAMES[k]})", ev)

    print()
    print("=" * 60)
    print(f"Phase 2 - Branch B (Human routing, alpha_B={alpha_B})")
    print("=" * 60)
    results["branch_b"] = {}
    for k in range(NUM_DISEASES):
        path = os.path.join(output_dir, "step6", f"lec_branch_b_disease_{k}_beta_{beta}_alphaB_{alpha_B}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            ev = evaluate_lec_result(data, alpha_B)
            results["branch_b"][k] = ev
            print_eval(f"disease_{k} ({DISEASE_NAMES[k]})", ev)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    p1_ok = results.get("phase_1", {}).get("controlled")
    a_controlled = sum(1 for v in results["branch_a"].values() if v["controlled"])
    a_total = len(results["branch_a"])
    b_controlled = sum(1 for v in results["branch_b"].values() if v["controlled"])
    b_total = len(results["branch_b"])

    print(f"  Phase 1:  {'PASS' if p1_ok else 'FAIL' if p1_ok is not None else 'N/A'} (mean_fdp={results.get('phase_1', {}).get('mean_fdp')} <= beta={beta})")
    print(f"  Branch A: {a_controlled}/{a_total} diseases controlled (mean_fdp <= alpha_A={alpha_A})")
    print(f"  Branch B: {b_controlled}/{b_total} diseases controlled (mean_fdp <= alpha_B={alpha_B})")

    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        eval_out = {
            "beta": beta,
            "alpha_A": alpha_A,
            "alpha_B": alpha_B,
            "phase_1": results.get("phase_1"),
            "branch_a": {str(k): v for k, v in results["branch_a"].items()},
            "branch_b": {str(k): v for k, v in results["branch_b"].items()},
        }
        json.dump(eval_out, f, indent=2)
    print(f"\n  Saved evaluation results to {eval_path}")


def main():
    p = argparse.ArgumentParser(description="Evaluate LEC calibration results")
    p.add_argument("--output_dir", default="result/json")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--alpha_A", type=float, default=0.1)
    p.add_argument("--alpha_B", type=float, default=0.1)
    args = p.parse_args()
    run(args.output_dir, args.beta, args.alpha_A, args.alpha_B)


if __name__ == "__main__":
    main()
