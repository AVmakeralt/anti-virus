"""
AdaptiveAV — Truly Adaptive ML Core
======================================
This replaces the fake "adaptive" classifier with a genuinely adaptive system:

1. FULL BACKPROPAGATION  — gradients flow through all layers (W1,W2,W3), not
   just a single bias tweak.

2. EXPERIENCE REPLAY BUFFER  — stores the last N labeled samples and replays
   a random mini-batch on every learn() call, preventing catastrophic forgetting.

3. CONCEPT DRIFT DETECTOR  — monitors rolling prediction error using Page-
   Hinkley test. When drift is detected, learning rate spikes and a full
   replay sweep is triggered.

4. FEATURE IMPORTANCE WEIGHTING  — each feature dimension has a learned
   weight updated via gradient signal magnitude. High-signal features are
   amplified; noisy ones are suppressed.

5. UNCERTAINTY QUANTIFICATION (Monte Carlo Dropout)  — stochastic forward
   passes with different dropout masks give a calibrated confidence interval,
   not just a raw sigmoid output.

6. DYNAMIC ENSEMBLE META-LEARNER  — learns *which* sub-model (Naive Bayes,
   NN, rule engine) to trust more for each sample type, based on historical
   accuracy per feature-cluster.

7. ADAPTIVE RULE WEIGHT TUNING  — heuristic rule weights are updated based
   on whether each rule correctly contributed to a final verdict (credit
   assignment via outcome).

8. CURRICULUM LEARNING  — hard examples (near the decision boundary) get
   repeated more often in replay; easy examples phase out over time.
"""

import math
import random
import pickle
import json
import time
import array
from pathlib import Path
from collections import deque, defaultdict, Counter
from typing import Optional

ADAPTIVEAV_DIR = Path.home() / ".adaptiveav"
ML_STATE_PATH  = ADAPTIVEAV_DIR / "adaptive_ml_state.pkl"

# ─────────────────────────────────────────────────────────────────
# Math helpers (pure Python, no deps)
# ─────────────────────────────────────────────────────────────────

def _sig(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def _relu(x: float) -> float:
    return max(0.0, x)

def _leaky_relu(x: float, a: float = 0.01) -> float:
    return x if x > 0 else a * x

def _drelu(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def _dleaky_relu(x: float, a: float = 0.01) -> float:
    return 1.0 if x > 0 else a

def _dsig(s: float) -> float:      # s is already sigmoid(x)
    return s * (1.0 - s)

def _mat_vec(W, x):
    return [sum(W[i][j] * x[j] for j in range(len(x))) + W[i][-1]  # last col = bias
            for i in range(len(W))]

def _softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]

def _clip_grad(g, clip=5.0):
    return max(-clip, min(clip, g))

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ─────────────────────────────────────────────────────────────────
# Experience Replay Buffer
# ─────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Prioritized experience replay.
    Priority = |error|  (higher error = harder example = replayed more often).
    Also tracks per-sample replay count to implement curriculum forgetting.
    """
    def __init__(self, maxlen: int = 2000):
        self.maxlen  = maxlen
        self._buf:   deque = deque(maxlen=maxlen)   # (x, label, priority, age)
        self._total  = 0

    def push(self, x: list, label: str, priority: float = 1.0):
        self._buf.append({
            "x":       x,
            "label":   label,
            "priority": max(0.01, priority),
            "age":     self._total,
            "replays": 0,
        })
        self._total += 1

    def sample(self, n: int, temperature: float = 1.0) -> list:
        """
        Sample n items.  Priority-weighted with temperature:
          temperature=1  → proportional to priority
          temperature→0  → uniform
          temperature>1  → focuses harder on high-priority
        """
        if not self._buf:
            return []
        items = list(self._buf)
        if len(items) <= n:
            for it in items: it["replays"] += 1
            return items

        # Compute sampling weights
        priors = [it["priority"] ** temperature for it in items]
        total  = sum(priors)
        probs  = [p / total for p in priors]

        # Weighted sampling without replacement
        chosen = []
        used   = set()
        for _ in range(n):
            r = random.random()
            cumsum = 0.0
            for i, p in enumerate(probs):
                if i in used:
                    continue
                cumsum += p / (1 - sum(probs[j] for j in used) + 1e-9)
                if r <= cumsum:
                    chosen.append(items[i])
                    used.add(i)
                    break
            else:
                # Fallback: pick any unchosen
                for i in range(len(items)):
                    if i not in used:
                        chosen.append(items[i])
                        used.add(i)
                        break
        for it in chosen:
            it["replays"] += 1
        return chosen

    def update_priority(self, x: list, new_priority: float):
        """Update priority of the most recently matching sample."""
        x_id = id(x)
        for item in self._buf:
            if item["x"] is x:
                item["priority"] = max(0.01, new_priority)
                return

    def __len__(self):
        return len(self._buf)

    def class_balance(self) -> dict:
        counts = Counter(it["label"] for it in self._buf)
        return dict(counts)


# ─────────────────────────────────────────────────────────────────
# Page-Hinkley Concept Drift Detector
# ─────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Page-Hinkley test for concept drift detection.
    Tracks running mean of prediction errors.
    When the cumulative deviation exceeds threshold → drift detected.
    """
    def __init__(self, threshold: float = 50.0, alpha: float = 0.005):
        self.threshold  = threshold
        self.alpha      = alpha   # allowed mean increase per step
        self._cumsum    = 0.0
        self._min_cumsum = 0.0
        self._n         = 0
        self._mean      = 0.0
        self.drift_count = 0
        self._error_window: deque = deque(maxlen=200)

    def update(self, error: float) -> bool:
        """
        Feed in abs(pred - truth).  Returns True if drift detected.
        """
        self._n += 1
        self._error_window.append(error)
        self._mean = (self._mean * (self._n - 1) + error) / self._n

        # Page-Hinkley statistic
        self._cumsum += error - self._mean - self.alpha
        self._min_cumsum = min(self._min_cumsum, self._cumsum)

        drift = (self._cumsum - self._min_cumsum) > self.threshold
        if drift:
            self._reset_after_drift()
            self.drift_count += 1
            return True
        return False

    def _reset_after_drift(self):
        self._cumsum = 0.0
        self._min_cumsum = 0.0
        # Keep mean estimate — only reset statistics

    def recent_error_rate(self) -> float:
        if not self._error_window:
            return 0.0
        return sum(self._error_window) / len(self._error_window)

    def stability(self) -> float:
        """0 = very unstable (drifting), 1 = stable."""
        return max(0.0, 1.0 - self.recent_error_rate())


# ─────────────────────────────────────────────────────────────────
# Adaptive Feature Importance Module
# ─────────────────────────────────────────────────────────────────

class FeatureImportanceTracker:
    """
    Tracks which features contribute most to correct predictions
    via accumulated gradient magnitude (|∂L/∂x_i|).
    Used to scale input features before NN forward pass.
    """
    def __init__(self, n_features: int):
        self.n = n_features
        # Accumulated |gradient| per feature
        self._accum_grad = [0.0] * n_features
        self._n_updates  = 0
        # Smoothed importance (EMA)
        self._importance = [1.0] * n_features
        self._ema_alpha  = 0.05

    def update(self, input_grads: list):
        """Feed in the gradient w.r.t. input layer."""
        if len(input_grads) != self.n:
            return
        self._n_updates += 1
        for i, g in enumerate(input_grads):
            # EMA update
            self._importance[i] = (
                (1 - self._ema_alpha) * self._importance[i] +
                self._ema_alpha * abs(g)
            )

    def scale_input(self, x: list) -> list:
        """Scale input by learned importance weights (softmax-normalized)."""
        # Softmax-normalize importances so they sum to n (preserve scale)
        m = max(self._importance)
        exp_imp = [math.exp(v - m) for v in self._importance]
        s = sum(exp_imp)
        weights = [v / s * self.n for v in exp_imp]
        return [x[i] * weights[i] for i in range(self.n)]

    def top_features(self, names: list, k: int = 5) -> list:
        """Return top-k most important feature names."""
        ranked = sorted(enumerate(self._importance), key=lambda t: t[1], reverse=True)
        return [(names[i] if i < len(names) else f"f{i}", imp) for i, imp in ranked[:k]]

    def state(self) -> dict:
        return {"importance": self._importance, "n_updates": self._n_updates}

    def load(self, d: dict):
        self._importance = d.get("importance", [1.0] * self.n)
        self._n_updates  = d.get("n_updates", 0)


# ─────────────────────────────────────────────────────────────────
# Truly Adaptive Neural Network (Full Backprop)
# ─────────────────────────────────────────────────────────────────

class AdaptiveNeuralNet:
    """
    3-hidden-layer neural network with:
      - Full backpropagation (all weights, all biases)
      - Adam optimizer (adaptive per-weight learning rates)
      - Monte Carlo Dropout for uncertainty estimation
      - Gradient clipping
      - L2 regularization

    Architecture: n_in → H1 → H2 → H3 → 1 (sigmoid)
    Default: 20 → 64 → 32 → 16 → 1
    """

    def __init__(self, n_in: int = 20,
                 hidden: tuple = (64, 32, 16),
                 lr: float = 0.01,
                 l2: float = 1e-4,
                 dropout_rate: float = 0.3):
        self.n_in        = n_in
        self.hidden      = hidden
        self.lr          = lr
        self.lr_init     = lr
        self.l2          = l2
        self.dropout_rate = dropout_rate
        self.n_trained   = 0

        # Adam state
        self._t = 0
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps   = 1e-8

        self._init_weights()

    def _init_weights(self):
        rng = random.Random(1337)
        layers = [self.n_in] + list(self.hidden) + [1]
        self.W = []  # W[l] is (out x in+1) matrix (last column = bias)
        self.mW = []  # Adam first moment
        self.vW = []  # Adam second moment

        for i in range(len(layers) - 1):
            fan_in  = layers[i]
            fan_out = layers[i + 1]
            # He initialization for ReLU layers
            std = math.sqrt(2.0 / fan_in)
            W = [[rng.gauss(0, std) for _ in range(fan_in + 1)]
                 for _ in range(fan_out)]
            self.W.append(W)
            self.mW.append([[0.0] * (fan_in + 1)] * fan_out)
            self.vW.append([[0.0] * (fan_in + 1)] * fan_out)

        self._z  = []  # pre-activations per layer
        self._a  = []  # activations per layer

    def _forward(self, x: list, training: bool = False) -> float:
        """Full forward pass. Returns output probability."""
        self._z = []
        self._a = [x]
        cur = x

        for l, W in enumerate(self.W):
            is_last = (l == len(self.W) - 1)
            # Append bias term
            cur_b = cur + [1.0]
            z = [sum(W[i][j] * cur_b[j] for j in range(len(cur_b)))
                 for i in range(len(W))]
            self._z.append(z)

            if is_last:
                a = [_sig(z[0])]
            else:
                a = [_leaky_relu(v) for v in z]
                # Monte Carlo Dropout (only during training OR inference with dropout)
                if training and self.dropout_rate > 0:
                    a = [0.0 if random.random() < self.dropout_rate else v / (1 - self.dropout_rate)
                         for v in a]
            self._a.append(a)
            cur = a

        return self._a[-1][0]

    def predict_proba(self, x: list, n_samples: int = 10) -> tuple[float, float]:
        """
        Monte Carlo Dropout inference.
        Returns (mean_probability, uncertainty_std).
        n_samples stochastic passes give a distribution → calibrated uncertainty.
        """
        samples = [self._forward(x, training=True) for _ in range(n_samples)]
        mean    = sum(samples) / n_samples
        if n_samples > 1:
            variance = sum((s - mean) ** 2 for s in samples) / (n_samples - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        return mean, std

    def _backprop(self, x: list, y: float) -> tuple[float, list]:
        """
        Full backpropagation.
        Returns (loss, input_gradients).
        """
        # Forward pass (deterministic for backprop)
        pred = self._forward(x, training=False)
        eps = 1e-7
        loss = -(y * math.log(pred + eps) + (1 - y) * math.log(1 - pred + eps))

        # Backprop
        n_layers = len(self.W)
        # dL/da_last
        delta = [pred - y]

        self._t += 1
        bc1 = 1 - self._beta1 ** self._t
        bc2 = 1 - self._beta2 ** self._t

        # Propagate backwards
        for l in reversed(range(n_layers)):
            is_last = (l == n_layers - 1)
            a_prev  = self._a[l]     # input to this layer
            z_l     = self._z[l]
            a_prev_b = a_prev + [1.0]  # with bias

            # Gradient w.r.t. pre-activation
            if is_last:
                dz = delta  # sigmoid derivative already absorbed in (pred - y)
            else:
                # LeakyReLU derivative
                dz = [delta[i] * _dleaky_relu(z_l[i]) for i in range(len(z_l))]

            # Update weights (Adam)
            W_l = self.W[l]
            for i in range(len(W_l)):
                for j in range(len(a_prev_b)):
                    g = _clip_grad(dz[i] * a_prev_b[j] + self.l2 * W_l[i][j])
                    # Adam update
                    self.mW[l][i] = [0.0] * len(a_prev_b)  # reinit if needed
                    old_m = self.mW[l][i][j] if j < len(self.mW[l][i]) else 0.0
                    old_v = self.vW[l][i][j] if j < len(self.vW[l][i]) else 0.0
                    new_m = self._beta1 * old_m + (1 - self._beta1) * g
                    new_v = self._beta2 * old_v + (1 - self._beta2) * g * g
                    # Reuse lists directly
                    try:
                        self.mW[l][i][j] = new_m
                        self.vW[l][i][j] = new_v
                    except IndexError:
                        pass
                    m_hat = new_m / bc1
                    v_hat = new_v / bc2
                    self.W[l][i][j] -= self.lr * m_hat / (math.sqrt(v_hat) + self._eps)

            # Propagate delta to previous layer
            if l > 0:
                new_delta = [0.0] * len(a_prev)
                for j in range(len(a_prev)):
                    for i in range(len(W_l)):
                        new_delta[j] += dz[i] * W_l[i][j]
                delta = new_delta

        # Input gradient (for feature importance)
        input_grad = [0.0] * len(x)
        W0 = self.W[0]
        dz0 = [delta[i] * _dleaky_relu(self._z[0][i]) for i in range(len(self._z[0]))]
        for j in range(len(x)):
            for i in range(len(W0)):
                input_grad[j] += dz0[i] * W0[i][j]

        return loss, input_grad

    def train_step(self, x: list, y: float) -> tuple[float, list]:
        """Single training step. Returns (loss, input_grads)."""
        loss, grads = self._backprop(x, y)
        self.n_trained += 1
        return loss, grads

    def decay_lr(self, factor: float = 0.995):
        self.lr = max(self.lr_init * 0.01, self.lr * factor)

    def spike_lr(self, factor: float = 5.0):
        """Temporarily spike LR on drift detection."""
        self.lr = min(self.lr_init * 3.0, self.lr * factor)

    def state_dict(self) -> dict:
        return {
            "W": self.W, "mW": self.mW, "vW": self.vW,
            "lr": self.lr, "lr_init": self.lr_init,
            "n_trained": self.n_trained, "_t": self._t,
            "hidden": self.hidden, "n_in": self.n_in,
            "dropout_rate": self.dropout_rate,
        }

    def load_state(self, d: dict):
        self.W        = d.get("W", self.W)
        self.mW       = d.get("mW", self.mW)
        self.vW       = d.get("vW", self.vW)
        self.lr       = d.get("lr", self.lr)
        self.lr_init  = d.get("lr_init", self.lr)
        self.n_trained = d.get("n_trained", 0)
        self._t       = d.get("_t", 0)


# ─────────────────────────────────────────────────────────────────
# Adaptive Naive Bayes with Forgetting
# ─────────────────────────────────────────────────────────────────

class AdaptiveNaiveBayes:
    """
    Gaussian Naive Bayes with:
    - Exponential forgetting (old samples lose weight over time)
    - Welford online mean/variance estimation
    - Per-class sample weighting
    """
    def __init__(self, n: int, forget_rate: float = 0.001):
        self.n           = n
        self.forget_rate = forget_rate
        self.classes     = ["malicious", "benign"]
        # Welford online stats: mean, M2 (sum of squared deviations), weight
        self._mean  = {c: [0.0] * n for c in self.classes}
        self._M2    = {c: [1e-6] * n for c in self.classes}
        self._W     = {c: 0.0 for c in self.classes}  # total weight
        self._count = {c: 0   for c in self.classes}
        self.n_trained = 0

    def _var(self, cls: str, i: int) -> float:
        w = max(self._W[cls], 1e-9)
        return max(1e-4, self._M2[cls][i] / w)

    def learn(self, x: list, label: str, weight: float = 1.0):
        """
        Online weighted update via Welford algorithm.
        Applied forgetting: old stats are slightly decayed.
        """
        cls = label
        if cls not in self.classes:
            return
        # Apply forgetting to existing stats
        decay = 1.0 - self.forget_rate
        self._W[cls] *= decay
        for i in range(self.n):
            self._M2[cls][i] *= decay

        # Welford update with weight
        w_old = self._W[cls]
        self._W[cls] += weight
        w_new = self._W[cls]

        for i in range(self.n):
            old_mean = self._mean[cls][i]
            self._mean[cls][i] += weight * (x[i] - old_mean) / w_new
            self._M2[cls][i]   += weight * (x[i] - old_mean) * (x[i] - self._mean[cls][i])

        self._count[cls] += 1
        self.n_trained += 1

    def predict_log_proba(self, x: list) -> dict:
        total_w = sum(max(self._W[c], 1e-9) for c in self.classes)
        log_probs = {}
        for cls in self.classes:
            log_prior = math.log(max(self._W[cls], 1e-9) / total_w)
            ll = 0.0
            for i in range(self.n):
                mean = self._mean[cls][i]
                var  = self._var(cls, i)
                ll  += -0.5 * math.log(2 * math.pi * var) - (x[i] - mean) ** 2 / (2 * var)
            log_probs[cls] = log_prior + ll
        return log_probs

    def predict_proba(self, x: list) -> float:
        """Returns P(malicious | x)."""
        lp = self.predict_log_proba(x)
        mx = max(lp.values())
        es = {k: math.exp(v - mx) for k, v in lp.items()}
        return es["malicious"] / sum(es.values())

    def state_dict(self) -> dict:
        return {
            "_mean": self._mean, "_M2": self._M2,
            "_W": self._W, "_count": self._count,
            "n_trained": self.n_trained,
        }

    def load_state(self, d: dict):
        self._mean     = d.get("_mean", self._mean)
        self._M2       = d.get("_M2", self._M2)
        self._W        = d.get("_W", self._W)
        self._count    = d.get("_count", self._count)
        self.n_trained = d.get("n_trained", 0)


# ─────────────────────────────────────────────────────────────────
# Adaptive Rule Weight Engine
# ─────────────────────────────────────────────────────────────────

class AdaptiveRuleEngine:
    """
    Rule-based heuristic engine whose weights are updated via
    credit assignment: if a rule fired and the final verdict was
    correct → reinforce; wrong → weaken.
    """
    RULES = [
        "packed_binary",
        "obfuscated_script",
        "high_entropy",
        "suspicious_strings",
        "network_binary",
        "signature_hit",
        "rare_extension",
        "pe_with_network",
        "large_base64_block",
        "cron_persistence",
    ]

    def __init__(self):
        # Initial weights (each rule contributes this many "points" when fired)
        self._weights = {r: 2.0 for r in self.RULES}
        self._weights["signature_hit"]    = 10.0
        self._weights["packed_binary"]    = 3.0
        self._weights["obfuscated_script"]= 2.5
        self._weights["suspicious_strings"] = 3.0
        self._weights["cron_persistence"] = 3.5

        # Credit tracking: (fires, correct_fires)
        self._fires   = {r: 0 for r in self.RULES}
        self._correct = {r: 0 for r in self.RULES}
        self._lr = 0.05

    def score(self, features: dict) -> tuple[float, list]:
        """
        Returns (raw_score, list_of_fired_rules).
        Dynamically converts any True boolean feature into a "rule" so that the
        rule engine can adapt to new indicators discovered at runtime.
        """
        fired = []
        score = 0.0

        # existing baseline rules
        if features.get("known_signature"):
            fired.append("signature_hit")
        if features.get("entropy_very_high") and features.get("is_binary"):
            fired.append("packed_binary")
        if features.get("entropy_very_high") and features.get("is_script"):
            fired.append("obfuscated_script")
        elif features.get("entropy_high"):
            fired.append("high_entropy")
        if features.get("suspicious_string_count", 0) > 0:
            fired.append("suspicious_strings")
        if features.get("has_network") and features.get("is_binary"):
            fired.append("network_binary")
        if features.get("is_pe") and features.get("has_network"):
            fired.append("pe_with_network")
        if features.get("hex_string_count", 0) > 20 and features.get("is_script"):
            fired.append("large_base64_block")
        ext = features.get("extension", "")
        if ext in (".scr", ".pif", ".com", ".hta", ".cpl"):
            fired.append("rare_extension")
        if features.get("suspicious_strings") and "crontab" in features.get("suspicious_strings", []):
            fired.append("cron_persistence")

        # dynamic boolean-feature rules
        for k, v in features.items():
            if isinstance(v, bool) and v and k not in fired:
                # add to weights if new
                if k not in self._weights:
                    self._weights[k] = 1.0
                    self._fires[k] = 0
                    self._correct[k] = 0
                fired.append(k)

        for r in fired:
            score += self._weights.get(r, 1.0)
            self._fires[r] += 1

        return score, fired

    def update(self, fired_rules: list, was_correct: bool):
        """
        Credit assignment: update rule weights based on outcome.
        Only updates rules that are in our known rule set.
        """
        for r in fired_rules:
            if r not in self._weights:
                continue  # Skip signature names / unknown rule identifiers
            if was_correct:
                self._correct[r] += 1
                self._weights[r] = min(15.0, self._weights[r] * (1 + self._lr * 0.5))
            else:
                self._weights[r] = max(0.1, self._weights[r] * (1 - self._lr))

        # Normalize weights to prevent runaway
        total = sum(self._weights.values())
        if total > 100:
            factor = 100 / total
            for r in self._weights:
                self._weights[r] *= factor

    def accuracy_per_rule(self) -> dict:
        # include any dynamically added rules (keys in _weights)
        return {
            r: self._correct.get(r, 0) / max(1, self._fires.get(r, 0))
            for r in self._weights.keys()
        }

    def state_dict(self) -> dict:
        return {"weights": self._weights, "fires": self._fires, "correct": self._correct}

    def load_state(self, d: dict):
        self._weights = d.get("weights", self._weights)
        self._fires   = d.get("fires", self._fires)
        self._correct = d.get("correct", self._correct)


# ─────────────────────────────────────────────────────────────────
# Dynamic Ensemble Meta-Learner
# ─────────────────────────────────────────────────────────────────

class MetaLearner:
    """
    Learns to weight the ensemble components (NB, NN, Rules) based on
    their historical accuracy per feature cluster.

    Feature clusters are defined by file type × entropy tier:
      cluster = (file_type, entropy_tier)
      file_type ∈ {binary, script, document, archive, other}
      entropy_tier ∈ {low, medium, high, very_high}
    """
    COMPONENTS = ["nb", "nn", "rules"]

    def __init__(self):
        # Per-cluster, per-component: (total_correct, total_seen)
        self._stats: dict[str, dict[str, list]] = defaultdict(
            lambda: {c: [0, 0] for c in self.COMPONENTS}
        )
        # Smoothed weights (EMA)
        self._weights: dict[str, dict[str, float]] = defaultdict(
            lambda: {c: 1.0 / len(self.COMPONENTS) for c in self.COMPONENTS}
        )
        self._ema = 0.1
        self.n_updates = 0

    def _cluster(self, features: dict) -> str:
        ext = features.get("extension", "")
        if features.get("is_binary"):
            ft = "binary"
        elif features.get("is_script"):
            ft = "script"
        elif ext in (".pdf", ".doc", ".docx", ".xls", ".xlsx"):
            ft = "document"
        elif ext in (".zip", ".rar", ".7z", ".tar", ".gz"):
            ft = "archive"
        else:
            ft = "other"

        e = features.get("entropy", 4.0)
        if e < 4.0:   et = "low"
        elif e < 6.0: et = "medium"
        elif e < 7.2: et = "high"
        else:          et = "very_high"

        return f"{ft}_{et}"

    def get_weights(self, features: dict) -> dict[str, float]:
        """Return weights for each component for this feature cluster."""
        cluster = self._cluster(features)
        w = self._weights[cluster]
        # Normalize
        total = sum(w.values())
        return {c: w[c] / total for c in self.COMPONENTS}

    def update(self, features: dict, component_preds: dict[str, float],
               true_label: int):
        """
        Update component weights based on who was right.
        component_preds: {"nb": float, "nn": float, "rules": float} (malicious prob)
        true_label: 0 or 1
        """
        cluster = self._cluster(features)
        for comp, pred in component_preds.items():
            correct = int((pred > 0.5) == bool(true_label))
            self._stats[cluster][comp][0] += correct
            self._stats[cluster][comp][1] += 1
            # Recompute weight via accuracy
            n = self._stats[cluster][comp][1]
            if n >= 3:
                acc = self._stats[cluster][comp][0] / n
                new_w = max(0.05, acc)
            else:
                new_w = 1.0 / len(self.COMPONENTS)
            self._weights[cluster][comp] = (
                (1 - self._ema) * self._weights[cluster][comp] + self._ema * new_w
            )
        self.n_updates += 1

    def state_dict(self) -> dict:
        return {
            "stats":   {k: v for k, v in self._stats.items()},
            "weights": {k: v for k, v in self._weights.items()},
            "n_updates": self.n_updates,
        }

    def load_state(self, d: dict):
        for k, v in d.get("stats", {}).items():
            self._stats[k] = v
        for k, v in d.get("weights", {}).items():
            self._weights[k] = v
        self.n_updates = d.get("n_updates", 0)


# ─────────────────────────────────────────────────────────────────
# Master Adaptive Classifier — integrates everything
# ─────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "entropy_norm", "entropy_high", "entropy_very_high",
    "unique_bytes_norm", "null_byte_ratio", "printable_ratio",
    "high_byte_ratio", "is_pe", "is_script",
    "sig_hit_count_norm", "known_signature",
    "sus_string_count_norm", "long_string_count_norm",
    "hex_string_count_norm", "has_network",
    "url_count_norm", "ip_count_norm",
    "section_entropy_var", "max_section_entropy",
    "file_size_norm",
]

class TrulyAdaptiveClassifier:
    """
    The main classifier. Brings together:
      - AdaptiveNaiveBayes       (fast, interpretable baseline)
      - AdaptiveNeuralNet        (full backprop, Adam, MC Dropout)
      - AdaptiveRuleEngine       (learned heuristic weights)
      - MetaLearner              (dynamic ensemble weighting)
      - ReplayBuffer             (prioritized experience replay)
      - DriftDetector            (concept drift detection)
      - FeatureImportanceTracker (learned input scaling)
    """

    def __init__(self, n_features: int = 20):
        self.n_features = n_features

        self.nb       = AdaptiveNaiveBayes(n_features)
        self.nn       = AdaptiveNeuralNet(n_features, hidden=(64, 32, 16))
        self.rules    = AdaptiveRuleEngine()
        self.meta     = MetaLearner()
        self.replay   = ReplayBuffer(maxlen=2000)
        self.drift    = DriftDetector(threshold=40.0)
        self.feat_imp = FeatureImportanceTracker(n_features)

        # Statistics
        self.n_learned     = 0
        self.n_correct     = 0
        self.n_drifts      = 0
        self._recent_accs: deque = deque(maxlen=100)

        # Replay batch size and frequency
        self._replay_batch = 32
        self._replay_every = 5   # replay every N learns

    def learn(self, x: list, label: str, features: dict = None,
              fired_rules: list = None, weight: float = 1.0):
        """
        Full online learning step.
        x            : feature vector (length n_features)
        label        : "malicious" | "benign"
        features     : raw feature dict (for meta-learner cluster)
        fired_rules  : which heuristic rules fired (for credit assignment)
        weight       : sample weight (higher = more important)
        """
        y = 1.0 if label == "malicious" else 0.0

        # 1. Get current prediction (before update, for error calculation)
        prev_pred, _ = self.predict(x, features)
        error = abs(prev_pred - y)

        # 2. Scale input by learned feature importance
        x_scaled = self.feat_imp.scale_input(x)

        # 3. Update Naive Bayes (with importance weighting)
        self.nb.learn(x_scaled, label, weight=weight)

        # 4. Full backprop through Neural Net
        loss, input_grads = self.nn.train_step(x_scaled, y)

        # 5. Update feature importance from gradient signal
        self.feat_imp.update(input_grads)

        # 6. Update rule weights (credit assignment)
        if fired_rules:
            correct = (prev_pred > 0.5) == bool(y)
            self.rules.update(fired_rules, correct)

        # 7. Concept drift detection
        drift = self.drift.update(error)
        if drift:
            self.n_drifts += 1
            self.nn.spike_lr(factor=4.0)
            # Trigger full replay on drift
            self._full_replay()

        # 8. Add to replay buffer (priority = prediction error)
        self.replay.push(x, label, priority=max(error, 0.05))

        # 9. Mini-batch replay (every N steps to prevent forgetting)
        self.n_learned += 1
        if self.n_learned % self._replay_every == 0:
            self._mini_replay()

        # 10. Update meta-learner
        if features:
            nb_p  = self.nb.predict_proba(x_scaled)
            nn_p  = self.nn._forward(x_scaled, training=False)
            rs, _ = self.rules.score(features)
            rule_p = min(rs / 8.0, 1.0)
            self.meta.update(features, {"nb": nb_p, "nn": nn_p, "rules": rule_p}, int(y))

        # 11. LR decay
        self.nn.decay_lr(factor=0.9995)

        # Track accuracy
        new_pred, _ = self.predict(x, features)
        correct = int((new_pred > 0.5) == bool(y))
        self._recent_accs.append(correct)

    def _mini_replay(self):
        """Replay a prioritized mini-batch."""
        batch = self.replay.sample(
            min(self._replay_batch, len(self.replay)),
            temperature=1.5  # prefer hard examples
        )
        for item in batch:
            x_s = self.feat_imp.scale_input(item["x"])
            y   = 1.0 if item["label"] == "malicious" else 0.0
            loss, grads = self.nn.train_step(x_s, y)
            self.feat_imp.update(grads)
            # Update priority based on new loss
            item["priority"] = max(0.01, min(10.0, loss))

    def _full_replay(self):
        """Full replay of entire buffer — triggered on drift detection."""
        all_items = list(self.replay._buf)
        random.shuffle(all_items)
        for item in all_items[:200]:  # Cap at 200 to avoid blocking
            x_s = self.feat_imp.scale_input(item["x"])
            y   = 1.0 if item["label"] == "malicious" else 0.0
            self.nn.train_step(x_s, y)
        self.nb = AdaptiveNaiveBayes(self.n_features)  # Reset NB stats post-drift
        for item in all_items:
            x_s = self.feat_imp.scale_input(item["x"])
            self.nb.learn(x_s, item["label"], weight=item.get("priority", 1.0))

    def predict(self, x: list, features: dict = None) -> tuple[float, dict]:
        """
        Full ensemble prediction.
        Returns (probability_malicious, details_dict).
        """
        x_scaled = self.feat_imp.scale_input(x)

        # Component predictions
        nb_p = self.nb.predict_proba(x_scaled)
        nn_p, nn_std = self.nn.predict_proba(x_scaled, n_samples=8)
        rule_score, fired = (0.0, [])
        if features:
            rule_score, fired = self.rules.score(features)
        rule_p = min(rule_score / 10.0, 1.0)

        # Dynamic ensemble weights from meta-learner
        if features and self.meta.n_updates >= 10:
            weights = self.meta.get_weights(features)
        else:
            # Default: equal weights until enough history
            n = self.n_learned
            nn_w = min(0.6, n / 100.0)
            nb_w = min(0.3, n / 150.0)
            rule_w = 0.1 + max(0.0, 0.3 - n / 100.0)
            total = nn_w + nb_w + rule_w
            weights = {
                "nn":    nn_w / total,
                "nb":    nb_w / total,
                "rules": rule_w / total,
            }

        # Signature override (always trust 100%)
        if features and features.get("known_signature"):
            combined = 0.99
        else:
            combined = (weights["nb"]    * nb_p +
                        weights["nn"]    * nn_p +
                        weights["rules"] * rule_p)

        # Calibrate confidence using MC Dropout uncertainty
        # High uncertainty → pull prediction toward 0.5 (less confident)
        uncertainty_discount = min(nn_std * 3.0, 0.3)
        calibrated = combined * (1 - uncertainty_discount) + 0.5 * uncertainty_discount

        details = {
            "nb_prob":          nb_p,
            "nn_prob":          nn_p,
            "nn_uncertainty":   nn_std,
            "rule_prob":        rule_p,
            "fired_rules":      fired,
            "ensemble_weights": weights,
            "calibrated_prob":  calibrated,
            "drift_detected":   self.drift.drift_count > 0,
        }

        return calibrated, details

    def recent_accuracy(self) -> float:
        if not self._recent_accs:
            return 0.5
        return sum(self._recent_accs) / len(self._recent_accs)

    def top_features(self) -> list:
        return self.feat_imp.top_features(FEATURE_NAMES, k=5)

    def state_dict(self) -> dict:
        return {
            "nb":       self.nb.state_dict(),
            "nn":       self.nn.state_dict(),
            "rules":    self.rules.state_dict(),
            "meta":     self.meta.state_dict(),
            "feat_imp": self.feat_imp.state(),
            "drift":    {"drift_count": self.drift.drift_count,
                         "n_drifts": self.n_drifts},
            "n_learned": self.n_learned,
            "n_correct": self.n_correct,
            "replay_len": len(self.replay),
        }

    def load_state(self, d: dict):
        if "nb"  in d: self.nb.load_state(d["nb"])
        if "nn"  in d: self.nn.load_state(d["nn"])
        if "rules" in d: self.rules.load_state(d["rules"])
        if "meta" in d: self.meta.load_state(d["meta"])
        if "feat_imp" in d: self.feat_imp.load(d["feat_imp"])
        dd = d.get("drift", {})
        self.drift.drift_count = dd.get("drift_count", 0)
        self.n_drifts  = dd.get("n_drifts", 0)
        self.n_learned = d.get("n_learned", 0)
        self.n_correct = d.get("n_correct", 0)

    @property
    def nb_trained(self) -> int:
        """Compat shim for code that checks nb_trained."""
        return self.nb.n_trained