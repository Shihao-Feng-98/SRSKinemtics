from IPython import embed
import numpy as np

from src.interval import Interval
from src.utils import K_EPS
from src.srs_kinematics import SRSKinematics, KineStatus, iiwa14


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
    test_feasible_arm_angle(iiwa_kine, 1000)