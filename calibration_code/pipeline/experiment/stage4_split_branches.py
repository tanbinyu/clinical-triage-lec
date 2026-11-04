def run(phase1_data, splits_phase1, lec_phase1):
    id_to_u1 = {sid: v["uncertainty"] for sid, v in phase1_data.items()}

    t_hat_by_epoch = {
        s["epoch"]: s["t_hat"] for s in lec_phase1["splits"]
    }

    branch_a_splits = []
    branch_b_splits = []

    for split in splits_phase1["splits"]:
        epoch = split["epoch"]
        t_hat = t_hat_by_epoch.get(epoch)

        if t_hat is None:
            continue

        a_calib = [sid for sid in split["calib"] if id_to_u1[sid] <= t_hat]
        a_test = [sid for sid in split["test"] if id_to_u1[sid] <= t_hat]
        b_calib = [sid for sid in split["calib"] if id_to_u1[sid] > t_hat]
        b_test = [sid for sid in split["test"] if id_to_u1[sid] > t_hat]

        branch_a_splits.append({"epoch": epoch, "calib": a_calib, "test": a_test})
        branch_b_splits.append({"epoch": epoch, "calib": b_calib, "test": b_test})

    return {"splits": branch_a_splits}, {"splits": branch_b_splits}
