"""数据加载模块：加载所有 JSON 文件并构建快速查找结构。"""
import json
import os

from pipeline.metrics.config import DATA_ROOT, NUM_DISEASES, DISEASE_NAMES


def _load(path):
    with open(path) as f:
        return json.load(f)


class DataLoader:
    """加载并缓存所有实验数据，提供快速查找接口。"""

    def __init__(self, beta=0.1, alpha=0.1, data_root=DATA_ROOT, input_root=None):
        self.beta = beta
        self.alpha = alpha
        self.data_root = os.fspath(DATA_ROOT if data_root is None else data_root)
        # A run may keep immutable step1 inputs outside its derived Step 2–6 root.
        self.input_root = os.fspath(input_root) if input_root is not None else self.data_root
        self._load_all()

    def _load_all(self):
        s1 = os.path.join(self.input_root, "step1")
        s2 = os.path.join(self.data_root, "step2")
        s3 = os.path.join(self.data_root, "step3")
        s4 = os.path.join(self.data_root, "step4")
        s5 = os.path.join(self.data_root, "step5")
        s6 = os.path.join(self.data_root, "step6")

        # ---- phase_1 ----
        self.phase1 = _load(os.path.join(s1, "phase_1.json"))

        # ---- splits_phase_1 ----
        self.splits_phase1 = _load(os.path.join(s2, "splits_phase_1.json"))

        # ---- lec_phase_1 (主实验) ----
        self.lec_phase1 = _load(os.path.join(s3, f"lec_phase_1_beta_{self.beta}.json"))
        self.lec_phase1_t_hats = [s["t_hat"] for s in self.lec_phase1["splits"]]

        # ---- splits_branch_a / b ----
        self.splits_branch_a = _load(os.path.join(s4, f"splits_branch_a_beta_{self.beta}.json"))
        self.splits_branch_b = _load(os.path.join(s4, f"splits_branch_b_beta_{self.beta}.json"))

        # ---- branch_a / b disease data ----
        self.branch_a = {}   # {k: {sample_id: {uncertainty, routing_correctness, ...}}}
        self.branch_b = {}
        for k in range(NUM_DISEASES):
            self.branch_a[k] = _load(os.path.join(s5, f"branch_a_disease_{k}.json"))
            self.branch_b[k] = _load(os.path.join(s5, f"branch_b_disease_{k}.json"))

        # ---- lec_branch_a / b (主实验) ----
        self.lec_branch_a = {}  # {k: lec_dict}
        self.lec_branch_a_t_hats = {}  # {k: [t_hat_per_split]}
        self.lec_branch_b = {}
        self.lec_branch_b_t_hats = {}
        for k in range(NUM_DISEASES):
            fa = os.path.join(s6, f"lec_branch_a_disease_{k}_beta_{self.beta}_alphaA_{self.alpha}.json")
            if os.path.exists(fa):
                d = _load(fa)
                self.lec_branch_a[k] = d
                self.lec_branch_a_t_hats[k] = [s["t_hat"] for s in d["splits"]]
            else:
                self.lec_branch_a[k] = None
                self.lec_branch_a_t_hats[k] = [None] * len(self.splits_phase1["splits"])

            fb = os.path.join(s6, f"lec_branch_b_disease_{k}_beta_{self.beta}_alphaB_{self.alpha}.json")
            if os.path.exists(fb):
                d = _load(fb)
                self.lec_branch_b[k] = d
                self.lec_branch_b_t_hats[k] = [s["t_hat"] for s in d["splits"]]
            else:
                self.lec_branch_b[k] = None
                self.lec_branch_b_t_hats[k] = [None] * len(self.splits_phase1["splits"])

        # ---- 构建快速查找字典 ----
        # phase1_lookup: {sample_id: (u1, predicted_disease, correctness)}
        self.phase1_lookup = {}
        for sid, v in self.phase1.items():
            self.phase1_lookup[sid] = (
                v["uncertainty"],
                v["predicted_disease"],
                v["correctness"],
            )

        # branch_a_lookup: {k: {sample_id: (uncertainty, routing_correctness, pred_correctness, correctness)}}
        self.branch_a_lookup = {}
        for k in range(NUM_DISEASES):
            self.branch_a_lookup[k] = {}
            for sid, v in self.branch_a[k].items():
                self.branch_a_lookup[k][sid] = (
                    v["uncertainty"],
                    v["routing_correctness"],
                    v["pred_correctness"],
                    v["correctness"],
                )

    def load_lec_phase1(self, beta):
        """加载指定 beta 的 lec_phase_1 文件。"""
        path = os.path.join(self.data_root, "step3", f"lec_phase_1_beta_{beta}.json")
        if os.path.exists(path):
            return _load(path)
        return None

    def load_lec_branch_a(self, k, alpha):
        """加载指定疾病 k 和 alpha 的 lec_branch_a 文件。"""
        path = os.path.join(
            self.data_root, "step6",
            f"lec_branch_a_disease_{k}_beta_{self.beta}_alphaA_{alpha}.json"
        )
        if os.path.exists(path):
            return _load(path)
        return None

    def load_lec_branch_b(self, k, alpha):
        """加载指定疾病 k 和 alpha 的 lec_branch_b 文件。"""
        path = os.path.join(
            self.data_root, "step6",
            f"lec_branch_b_disease_{k}_beta_{self.beta}_alphaB_{alpha}.json"
        )
        if os.path.exists(path):
            return _load(path)
        return None
