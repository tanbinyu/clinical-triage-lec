"""LEC calibration primitives shared by every experiment stage."""


def lec_calibrate(calib_ids, id_to_uncertainty, id_to_correctness, alpha):
    # t_hat = max{t in Lambda : sum_i l_i(t) <= -1}, where
    # l_i(t) = 1{u_i <= t, correctness_i = 0} - alpha * 1{u_i <= t}.
    # sum_i l_i(t) = Nerr(t) - alpha * Nacc(t) is not monotone in t, so the
    # feasible set is not an interval and every candidate has to be scanned.
    if not calib_ids:
        return None

    samples = sorted(
        (id_to_uncertainty[sid], id_to_correctness[sid]) for sid in calib_ids
    )

    t_hat = None
    errors = 0

    for i, (uncertainty, correctness) in enumerate(samples):
        errors += int(correctness == 0)

        # Acceptance uses u_i <= t, so tied scores are decided as one group.
        if i + 1 < len(samples) and samples[i + 1][0] == uncertainty:
            continue

        if errors - alpha * (i + 1) <= -1:
            t_hat = uncertainty

    return t_hat


def lec_evaluate(test_ids, id_to_uncertainty, id_to_correctness, t_hat):
    selected_num = 0
    error_num = 0

    for sid in test_ids:
        uncertainty = id_to_uncertainty[sid]
        correctness = id_to_correctness[sid]

        if uncertainty <= t_hat:
            selected_num += 1
            if correctness == 0:
                error_num += 1

    return selected_num, error_num
