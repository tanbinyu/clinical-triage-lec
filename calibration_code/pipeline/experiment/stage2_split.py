import random
from collections import defaultdict


def run(phase1_data, num_splits, split_ratio):
    dataset_groups = defaultdict(list)
    for sample_id, info in phase1_data.items():
        dataset_groups[info["source_dataset"]].append(sample_id)

    splits = []
    for epoch in range(num_splits):
        random.seed(epoch)
        calib_ids = []
        test_ids = []
        for dataset_name in sorted(dataset_groups.keys()):
            ids = dataset_groups[dataset_name].copy()
            random.shuffle(ids)
            n_calib = int(len(ids) * split_ratio)
            calib_ids.extend(ids[:n_calib])
            test_ids.extend(ids[n_calib:])
        splits.append({"epoch": epoch, "calib": calib_ids, "test": test_ids})

    return {"splits": splits}
