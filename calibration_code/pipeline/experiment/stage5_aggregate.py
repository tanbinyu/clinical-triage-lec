import json
import os
from collections import defaultdict

from pipeline.core.config import DISEASE_TO_DATASETS, NUM_DISEASES


def run(phase1_data, phase2_dir, output_dir, lambda_1, lambda_2):
    phase2_all = {}
    for _, datasets in DISEASE_TO_DATASETS.items():
        for ds_name in datasets:
            path = os.path.join(phase2_dir, f"phase_2_{ds_name}.json")
            if not os.path.exists(path):
                print(f"[WARNING] Missing phase_2 file: {path}, skipping dataset {ds_name}")
                continue
            with open(path) as f:
                phase2_all[ds_name] = json.load(f)

    branch_a_by_disease = defaultdict(dict)
    branch_b_by_disease = defaultdict(dict)

    for sample_id, p1 in phase1_data.items():
        source_dataset = p1["source_dataset"]
        predicted_k = p1["predicted_disease"]
        true_k = p1["true_disease"]

        if source_dataset not in phase2_all:
            continue
        p2_dataset = phase2_all[source_dataset]
        if sample_id not in p2_dataset:
            continue
        p2 = p2_dataset[sample_id]

        u1 = p1["uncertainty"]
        u2 = p2["uncertainty"]
        routing_correctness = int(predicted_k == true_k)
        pred_correctness = int(p2["correctness"])
        correctness = int(routing_correctness == 1 and pred_correctness == 1)

        branch_a_by_disease[predicted_k][sample_id] = {
            "uncertainty": round(lambda_1 * u1 + lambda_2 * u2, 6),
            "routing_correctness": routing_correctness,
            "pred_correctness": pred_correctness,
            "correctness": correctness,
        }

        branch_b_by_disease[true_k][sample_id] = {
            "uncertainty": u2,
            "correctness": pred_correctness,
        }

    for k in range(NUM_DISEASES):
        a_path = os.path.join(output_dir, f"branch_a_disease_{k}.json")
        b_path = os.path.join(output_dir, f"branch_b_disease_{k}.json")

        with open(a_path, "w") as f:
            json.dump(dict(branch_a_by_disease.get(k, {})), f, indent=2)
        with open(b_path, "w") as f:
            json.dump(dict(branch_b_by_disease.get(k, {})), f, indent=2)

    print(f"[Step 5] Saved branch_a/b_disease_{{0..{NUM_DISEASES-1}}}.json to {output_dir}")
