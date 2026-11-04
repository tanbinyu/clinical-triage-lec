import json
import os

from pipeline.core.config import NUM_DISEASES
from pipeline.core.lec import lec_calibrate, lec_evaluate


def _calibrate_branch(disease_data, splits, alpha):
    id_to_uncertainty = {sid: v["uncertainty"] for sid, v in disease_data.items()}
    id_to_correctness = {sid: int(v["correctness"]) for sid, v in disease_data.items()}
    disease_sample_ids = set(disease_data.keys())

    if not disease_sample_ids:
        return None

    total_correct = sum(id_to_correctness.values())
    base_error_rate = 1 - total_correct / len(id_to_correctness)

    result_splits = []
    for split in splits["splits"]:
        epoch = split["epoch"]
        calib_ids = [sid for sid in split["calib"] if sid in disease_sample_ids]
        test_ids = [sid for sid in split["test"] if sid in disease_sample_ids]

        total_num = len(test_ids)

        if not calib_ids:
            result_splits.append({
                "epoch": epoch,
                "t_hat": None,
                "total_num": total_num,
                "selected_num": 0,
                "error_num": 0,
            })
            continue

        t_hat = lec_calibrate(calib_ids, id_to_uncertainty, id_to_correctness, alpha)

        if t_hat is None:
            result_splits.append({
                "epoch": epoch,
                "t_hat": None,
                "total_num": total_num,
                "selected_num": 0,
                "error_num": 0,
            })
            continue

        selected_num, error_num = lec_evaluate(
            test_ids, id_to_uncertainty, id_to_correctness, t_hat
        )
        result_splits.append({
            "epoch": epoch,
            "t_hat": t_hat,
            "total_num": total_num,
            "selected_num": selected_num,
            "error_num": error_num,
        })

    return {
        "alpha": alpha,
        "base_error_rate": round(base_error_rate, 4),
        "splits": result_splits,
    }


def run(data_dir, output_dir, splits_a, splits_b, alpha_A, alpha_B, beta):
    for k in range(NUM_DISEASES):
        a_data_path = os.path.join(data_dir, f"branch_a_disease_{k}.json")
        if os.path.exists(a_data_path):
            with open(a_data_path) as f:
                a_data = json.load(f)
            result_a = _calibrate_branch(a_data, splits_a, alpha_A)
            if result_a is not None:
                a_out = os.path.join(output_dir, f"lec_branch_a_disease_{k}_beta_{beta}_alphaA_{alpha_A}.json")
                with open(a_out, "w") as f:
                    json.dump(result_a, f, indent=2)

        b_data_path = os.path.join(data_dir, f"branch_b_disease_{k}.json")
        if os.path.exists(b_data_path):
            with open(b_data_path) as f:
                b_data = json.load(f)
            result_b = _calibrate_branch(b_data, splits_b, alpha_B)
            if result_b is not None:
                b_out = os.path.join(output_dir, f"lec_branch_b_disease_{k}_beta_{beta}_alphaB_{alpha_B}.json")
                with open(b_out, "w") as f:
                    json.dump(result_b, f, indent=2)

    print(f"[Step 6] Saved lec_branch_a/b_disease_{{0..{NUM_DISEASES-1}}}.json to {output_dir}")
