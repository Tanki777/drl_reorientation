"""

The reward function differs depending on the training phase and whether the
safety filter is applied during training.

┌──────────────────────────────────────────────────────────────────────────┐
│  use_safety_filter = 0  →  Phase 1 (no safety filter, no KOZ)         │
│  use_safety_filter = 1  →  Phase 2, safety filter in simulation only  │
│  use_safety_filter = 2  →  Phase 2, safety filter applied in training │
│                                                                        │
│  KOZ penalty (r5) is always computed but effectively ≈ 0 in Phase 1   │
│  because the KOZ half-angle is 0° → θ_margin is large.                │
│                                                                        │
│  Action-divergence penalty (r6) is added ONLY when                     │
│  use_safety_filter = 2  (safety filter applied during training).       │
└──────────────────────────────────────────────────────────────────────────┘

─────────────────────────────────────────────────────────────────────────────
Phase 1  (use_safety_filter = 0)   — no keep-out zone, no safety filter
─────────────────────────────────────────────────────────────────────────────

    R_phase1 = r1 + r3 + r4 + r5          (r5 ≈ 0 because no KOZ)

─────────────────────────────────────────────────────────────────────────────
Phase 2  (use_safety_filter = 1)   — KOZ active, safety filter in sim only
─────────────────────────────────────────────────────────────────────────────

    R_phase2_sim = r1 + r3 + r4 + r5

─────────────────────────────────────────────────────────────────────────────
Phase 2  (use_safety_filter = 2)   — KOZ active, safety filter in training
─────────────────────────────────────────────────────────────────────────────

    R_phase2_train = r1 + r3 + r4 + r5 + r6

─────────────────────────────────────────────────────────────────────────────
Component definitions
─────────────────────────────────────────────────────────────────────────────

    r1 (Attitude Error Reduction):
        r1 = φ_prev − φ_current
        where φ = 2 · arccos(q0) · (180 / π)   [deg]
        Positive when the pointing error decreases between steps.

    r3 (Accuracy Bonus):
        r3 =  +0.01   if φ_current ≤ 0.25°
        r3 =  −0.01   otherwise

    r4 (Torque Penalty):
        r4 = −(|τ₁| + |τ₂| + |τ₃|)

    r5 (Keep-Out Zone Penalty):
        r5 = −1.0                   if θ_margin ≤ 0   (inside KOZ)
        r5 = −exp(−66 · θ_margin)   if θ_margin > 0   (near KOZ boundary)
        where θ_margin is the angular margin [rad] to the KOZ boundary.
        In Phase 1 (KOZ half-angle = 0°), θ_margin is large → r5 ≈ 0.

    r6 (Safety-Filter Action Divergence Penalty)  — only when use_safety_filter = 2:
        r6 = −(|u_safe₁ − u_agent₁| + |u_safe₂ − u_agent₂| + |u_safe₃ − u_agent₃|)
        Penalises the agent when its proposed action differs from the
        safety-filtered action, encouraging the agent to learn safe behaviour.
"""

import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 reward  (no safety filter, no KOZ)
# ─────────────────────────────────────────────────────────────────────────────
def compute_reward_phase1(q0_current, q0_prev, torque, margin_koz):
    """
    Phase 1 reward — no keep-out zone, no safety filter.

    R = r1 + r3 + r4 + r5   (r5 ≈ 0 because KOZ half-angle is 0°)

    Args:
        q0_current  (float): Scalar part of the current quaternion.
        q0_prev     (float): Scalar part of the previous-step quaternion.
        torque      (array): Applied torques [τ1, τ2, τ3] (Nm).
        margin_koz  (float): Angular margin to KOZ boundary (rad).

    Returns:
        float: Total Phase 1 reward.
    """
    return _base_reward(q0_current, q0_prev, torque, margin_koz)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 reward  — safety filter in simulation only (use_safety_filter = 1)
# ─────────────────────────────────────────────────────────────────────────────
def compute_reward_phase2_sim(q0_current, q0_prev, torque, margin_koz):
    """
    Phase 2 reward — safety filter applied in simulation, KOZ penalty active.

    R = r1 + r3 + r4 + r5

    Args:
        q0_current  (float): Scalar part of the current quaternion.
        q0_prev     (float): Scalar part of the previous-step quaternion.
        torque      (array): Applied torques [τ1, τ2, τ3] (Nm).
        margin_koz  (float): Angular margin to KOZ boundary (rad).

    Returns:
        float: Total Phase 2 (sim) reward.
    """
    return _base_reward(q0_current, q0_prev, torque, margin_koz)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 reward  — safety filter applied during training (use_safety_filter = 2)
# ─────────────────────────────────────────────────────────────────────────────
def compute_reward_phase2_train(q0_current, q0_prev, torque, margin_koz,
                                agent_action, safe_action):
    """
    Phase 2 reward — safety filter applied during training.
    Includes action-divergence penalty (r6).

    R = r1 + r3 + r4 + r5 + r6

    Args:
        q0_current   (float): Scalar part of the current quaternion.
        q0_prev      (float): Scalar part of the previous-step quaternion.
        torque       (array): Applied torques [τ1, τ2, τ3] (Nm).
        margin_koz   (float): Angular margin to KOZ boundary (rad).
        agent_action (array): Action proposed by the agent [u1, u2, u3].
        safe_action  (array): Action after safety filtering [u1, u2, u3].

    Returns:
        float: Total Phase 2 (training) reward.
    """
    base = _base_reward(q0_current, q0_prev, torque, margin_koz)

    # --- r6: Safety-filter action divergence penalty -------------------------
    r6 = -(abs(safe_action[0] - agent_action[0])
          + abs(safe_action[1] - agent_action[1])
          + abs(safe_action[2] - agent_action[2]))

    return base + r6


# ─────────────────────────────────────────────────────────────────────────────
# Convenience dispatcher (mirrors the environment logic)
# ─────────────────────────────────────────────────────────────────────────────
def compute_reward(q0_current, q0_prev, torque, margin_koz=0.0,
                   agent_action=None, safe_action=None, use_safety_filter=0):
    """
    Dispatch to the correct reward function based on the safety-filter mode.

    Args:
        q0_current       (float): Scalar part of the current quaternion.
        q0_prev          (float): Scalar part of the previous-step quaternion.
        torque           (array): Applied torques [τ1, τ2, τ3] (Nm).
        margin_koz       (float): Angular margin to KOZ boundary (rad).
        agent_action     (array): Action proposed by the agent (needed for mode 2).
        safe_action      (array): Action after safety filter (needed for mode 2).
        use_safety_filter  (int): 0 = off (Phase 1),
                                  1 = applied during simulation (Phase 2),
                                  2 = applied during training  (Phase 2).

    Returns:
        float: Total reward.
    """
    if use_safety_filter == 0:
        # Phase 1 — no safety filter, r5 ≈ 0
        return compute_reward_phase1(q0_current, q0_prev, torque, margin_koz)
    elif use_safety_filter == 1:
        # Phase 2 — safety filter in simulation only
        return compute_reward_phase2_sim(q0_current, q0_prev, torque, margin_koz)
    else:
        # Phase 2 — safety filter in training, includes r6
        return compute_reward_phase2_train(q0_current, q0_prev, torque, margin_koz,
                                           agent_action, safe_action)


# ─────────────────────────────────────────────────────────────────────────────
# Shared base reward  (r1 + r3 + r4 + r5)
# ─────────────────────────────────────────────────────────────────────────────
def _base_reward(q0_current, q0_prev, torque, margin_koz):
    """Return the base reward components shared by all modes."""
    # Clamp q0 values to [-1, 1] to prevent arccos domain errors
    q0_current = np.clip(q0_current, -1.0, 1.0)
    q0_prev    = np.clip(q0_prev,    -1.0, 1.0)

    # Attitude errors in degrees
    phi_current = 2.0 * math.acos(q0_current) * 180.0 / np.pi
    phi_prev    = 2.0 * math.acos(q0_prev)    * 180.0 / np.pi

    # --- r1: Attitude error reduction (positive when error decreases) --------
    r1 = phi_prev - phi_current

    # --- r3: Accuracy bonus -------------------------------------------------
    r3 = 0.01 if phi_current <= 0.25 else -0.01

    # --- r4: Torque penalty --------------------------------------------------
    r4 = -(abs(torque[0]) + abs(torque[1]) + abs(torque[2]))

    # --- r5: Keep-out zone penalty -------------------------------------------
    if margin_koz <= 0.0:
        r5 = -1.0                            # inside the KOZ
    else:
        r5 = -math.exp(-66.0 * margin_koz)   # exponential decay near boundary

    return r1 + r3 + r4 + r5


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q0_current = np.cos(np.deg2rad(10) / 2)   # 10° attitude error
    q0_prev    = np.cos(np.deg2rad(12) / 2)   # 12° previous error
    torque     = np.array([0.01, 0.02, 0.0])
    margin_koz = 0.15                          # ~8.6° margin (rad)
    agent_act  = np.array([0.8, 0.5, 0.1])
    safe_act   = np.array([0.6, 0.4, 0.1])

    print("=== Phase 1 (use_safety_filter=0, no KOZ) ===")
    r_p1 = compute_reward(q0_current, q0_prev, torque, margin_koz, use_safety_filter=0)
    print(f"  R = r1 + r3 + r4 + r5  →  {r_p1:.4f}")

    print("\n=== Phase 2 — safety filter in simulation (use_safety_filter=1) ===")
    r_p2_sim = compute_reward(q0_current, q0_prev, torque, margin_koz, use_safety_filter=1)
    print(f"  R = r1 + r3 + r4 + r5  →  {r_p2_sim:.4f}")

    print("\n=== Phase 2 — safety filter in training (use_safety_filter=2) ===")
    r_p2_train = compute_reward(q0_current, q0_prev, torque, margin_koz,
                                agent_act, safe_act, use_safety_filter=2)
    print(f"  R = r1 + r3 + r4 + r5 + r6  →  {r_p2_train:.4f}")