from pipeline.core.lec import lec_calibrate, lec_evaluate


def run(phase1_data, splits, beta):
    id_to_uncertainty = {sid: v["uncertainty"] for sid, v in phase1_data.items()}
    id_to_correctness = {sid: int(v["correctness"]) for sid, v in phase1_data.items()}

    total_correct = sum(id_to_correctness.values())
    base_error_rate = 1 - total_correct / len(id_to_correctness)

    result_splits = []
    for split in splits["splits"]:
        epoch = split["epoch"]
        calib_ids = split["calib"]
        test_ids = split["test"]

        t_hat = lec_calibrate(calib_ids, id_to_uncertainty, id_to_correctness, beta)

        total_num = len(test_ids)

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
        "beta": beta,
        "base_error_rate": round(base_error_rate, 4),
        "splits": result_splits,
    }
