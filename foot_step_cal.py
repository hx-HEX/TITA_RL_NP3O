import math

# 你的默认值
theta1 = 0.858   # hip default (rad)
theta2 = -1.755  # knee default (rad)
l1 = 0.2
l2 = 0.2

A1 = l1*math.cos(theta1) + l2*math.cos(theta1 + theta2)
A2 = l2*math.cos(theta1 + theta2)

def est_clearance(hip_amp, knee_amp):
    return A1*abs(hip_amp) + A2*abs(knee_amp)

# 例子：给定目标 h_clear，若固定 hip_amp 求最小 knee_amp
def min_knee_for_clearance(h_clear, hip_amp):
    rem = h_clear - A1*abs(hip_amp)
    if rem <= 0:
        return 0.0
    return rem / A2

# 检查关节极限（填入你的 URDF 极限）
def check_limits(theta_default, amp, low_limit, high_limit):
    return (theta_default - amp >= low_limit) and (theta_default + amp <= high_limit)

# demo:
h_clear = 0.08
hip_try = 0.10
knee_needed = min_knee_for_clearance(h_clear, hip_try)
print(f"Given hip_amp={hip_try:.3f} rad => knee_amp needed ≈ {knee_needed:.3f} rad")
print("Estimated clearance:", est_clearance(hip_try, knee_needed))

