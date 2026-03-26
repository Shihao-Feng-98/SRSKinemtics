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



if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    
    iiwa_kine = SRSKinematics(iiwa14())

    test_general_ik(iiwa_kine)
    test_shoulder_singular(iiwa_kine)
    test_elbow_singular(iiwa_kine)
    test_wrist_singular(iiwa_kine)
