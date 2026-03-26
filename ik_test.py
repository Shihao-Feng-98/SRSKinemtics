from IPython import embed
import numpy as np

from src.interval import Interval
from src.utils import K_EPS_SMALL, K_EPS, K_EPS_LARGE
from src.srs_kinematics import SRSKinematics, KineStatus, iiwa14


def test_ik(kine, qpos_seed) -> bool:
    res_psi, ref_psi = kine.calc_arm_angle(qpos_seed)
    if res_psi != KineStatus.OK:
        print(f"Calc arm angle failed! qpos_seed: {qpos_seed}")
        return False
    
    pose = kine.get_fk(qpos_seed)
    srs_cfg = SRSKinematics.Config.from_qpos(qpos_seed, ref_psi)

    res_ik, qpos = kine.get_ik(pose, srs_cfg, qpos_seed)
    if res_ik != KineStatus.OK:
        print(f"IK failed! qpos_seed: {qpos_seed}")
        return False
    elif not np.allclose(qpos_seed, qpos, atol=K_EPS_LARGE):
        print(f"IK solution mismatch! qpos_seed: {qpos_seed}, qpos: {qpos}")
        return False
    
    return True

def test_general_ik(kine, n: int = 1e4) -> bool:
    print("Start check general ik...")
    
    qpos_sample_array = np.random.uniform(kine.params.lb, 
                                          kine.params.ub, 
                                          (int(n), 7))
    for i, qpos_seed in enumerate(qpos_sample_array):
        if not test_ik(kine, qpos_seed):
            return False
        if i % 1000 == 0:
            print(f"Checked {i} samples...")
    print("Finish check general ik finished...")
    return True

def test_shoulder_singular(kine, n: int = 1e3) -> bool:
    print("Start check shoulder singular ik...")
    
    qpos_sample_array = np.random.uniform(kine.params.lb, 
                                          kine.params.ub, 
                                          (int(n), 7))
    for i, qpos_seed in enumerate(qpos_sample_array):
        if abs(qpos_seed[3]) < K_EPS:
            return True
        qpos_seed[1] = np.sign(qpos_seed[1]) * K_EPS_SMALL
        if not test_ik(kine, qpos_seed):
            return False
        if i % 100 == 0:
            print(f"Checked {i} samples...")
    print("Finish check shoulder singular ik...")
    return True

def test_elbow_singular(kine, n: int = 1e3) -> bool:
    print("Start check elbow singular ik...")
    
    qpos_sample_array = np.random.uniform(kine.params.lb, 
                                          kine.params.ub, 
                                          (int(n), 7))
    for i, qpos_seed in enumerate(qpos_sample_array):
        if abs(qpos_seed[1]) < K_EPS:
            return True
        if abs(qpos_seed[5]) < K_EPS:
            return True
        qpos_seed[3] = np.sign(qpos_seed[3]) * K_EPS_SMALL
        if not test_ik(kine, qpos_seed):
            return False
        if i % 100 == 0:
            print(f"Checked {i} samples...")
    print("Finish check elbow singular ik...")
    return True

def test_wrist_singular(kine, n: int = 1e3) -> bool:
    print("Start check wrist singular ik...")
    
    qpos_sample_array = np.random.uniform(kine.params.lb, 
                                          kine.params.ub, 
                                          (int(n), 7))
    for i, qpos_seed in enumerate(qpos_sample_array):
        if abs(qpos_seed[3]) < K_EPS:
            return True
        qpos_seed[5] = np.sign(qpos_seed[5]) * K_EPS_SMALL
        if not test_ik(kine, qpos_seed):
            return False
        if i % 100 == 0:
            print(f"Checked {i} samples...")
    print("Finish check wrist singular ik...")
    return True

def _test_feasible_arm_angle(kine, qpos_seed, debug=False) -> bool:
    pose = kine.get_fk(qpos_seed)
    srs_cfg = SRSKinematics.Config.from_qpos(qpos_seed)
    res, psi_intervals = kine.calc_feasible_arm_angle_intervals(pose, srs_cfg, debug)
    if res != KineStatus.OK:
        print(f"Calc feasible arm angle intervals failed! qpos_seed: {qpos_seed}")
        return False
    elif psi_intervals.is_empty():
        print(f"No feasible arm angle! qpos_seed: {qpos_seed}")
        embed(banner1="100")
        return False

    valid_psi_list = []
    for interval in psi_intervals.intervals:
        step = 0.05
        if interval.length() < step:
            valid_psi_list.extend(interval.sample_uniform_by_n(5, False))
        else: 
            valid_psi_list.extend(interval.sample_uniform_by_step(step, False))
    # check trues
    for psi in valid_psi_list:
        srs_cfg.arm_angle = psi
        res_ik, qpos = kine.get_ik(pose, srs_cfg, qpos_seed)
        if res_ik != KineStatus.OK:
            print(f"IK failed! qpos_seed: {qpos_seed}, psi: {psi}")
            embed(banner1="116")
            return False
        pose_ = kine.get_fk(qpos)
        if not np.allclose(pose_, pose, atol=K_EPS):
            print(f"IK solution mismatch! pose: {pose}, pose_: {pose_}")
            embed(banner1="121")
            return False

    invalid_psi_list = []
    psi_intervals2 = psi_intervals.complement(Interval(-np.pi, np.pi))
    for interval in psi_intervals2.intervals:
        step = 0.1
        if interval.length() < step:
            invalid_psi_list.extend(interval.sample_uniform_by_n(4, False))
        else: 
            invalid_psi_list.extend(interval.sample_uniform_by_step(step, False))
    # check false
    for psi in invalid_psi_list:
        srs_cfg.arm_angle = psi
        res_ik, qpos = kine.get_ik(pose, srs_cfg, qpos_seed, verbose=False)
        if res_ik == KineStatus.OK:
            print(f"IK should fail but success! qpos_seed: {qpos_seed}, psi: {psi}")
            embed(banner1="138")
            return False
    return True

def test_feasible_arm_angle(kine, n: int = 1e3):
    print("Start check feasible arm angle...")
    
    qpos_sample_array = np.random.uniform(kine.params.lb, 
                                          kine.params.ub, 
                                          (int(n), 7))
    for i, qpos_seed in enumerate(qpos_sample_array):
        if abs(qpos_seed[1]) < K_EPS:
            continue
        if abs(qpos_seed[3]) < K_EPS:
            continue
        if abs(qpos_seed[5]) < K_EPS:
            continue
        if not _test_feasible_arm_angle(kine, qpos_seed):
            return False
        if i % 100 == 0:
            print(f"Checked {i} samples...")
    print("Finish check feasible arm angle...")
    return True


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    
    iiwa_kine = SRSKinematics(iiwa14())

    # test_general_ik(iiwa_kine)
    # test_shoulder_singular(iiwa_kine)
    # test_elbow_singular(iiwa_kine)
    # test_wrist_singular(iiwa_kine)
    # test_feasible_arm_angle(iiwa_kine)

    test_feasible_arm_angle(iiwa_kine, 10000)
