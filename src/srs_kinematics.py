from __future__ import annotations

from dataclasses import dataclass, field
from IPython import embed
import numpy as np
from enum import Enum

from .utils import *
from .interval import *


class KineStatus(Enum):
    OK = 1
    
    OUT_OF_WORKSPACE = -1
    OUT_OF_JOINT_LIMITS = -2
    IK_NOT_CONTINUE = -3

    UNKNOWN = -100


@dataclass
class SRSParams:
    d_bs: float
    d_se: float
    d_ew: float
    d_wt: float
    lb_deg: tuple[float, ...]
    ub_deg: tuple[float, ...]
    mdh: np.ndarray = field(init=False)
    lb: np.ndarray = field(init=False)
    ub: np.ndarray = field(init=False)

    def __post_init__(self):
        if len(self.lb_deg) != 7 or len(self.ub_deg) != 7:
            raise ValueError("lb_deg and ub_deg must each contain 7 joint limits")

        # a(i-1) alpha(i-1) d(i) theta(i)
        self.mdh = np.array([[0.0,  0.0,       self.d_bs, 0.0],
                                    [0.0, -np.pi / 2,       0.0, 0.0],
                                    [0.0,  np.pi / 2, self.d_se, 0.0],
                                    [0.0, -np.pi / 2,       0.0, 0.0],
                                    [0.0,  np.pi / 2, self.d_ew, 0.0],
                                    [0.0, -np.pi / 2,       0.0, 0.0],
                                    [0.0,  np.pi / 2, self.d_wt, 0.0]])

        self.lb = np.deg2rad(self.lb_deg)
        self.ub = np.deg2rad(self.ub_deg)

@dataclass
class iiwa14(SRSParams):
    d_bs: float = 0.36
    d_se: float = 0.42
    d_ew: float = 0.4
    d_wt: float = 0.081
    lb_deg: tuple[float, ...] = (-170, -120, -170, -120, -170, -120, -175)
    ub_deg: tuple[float, ...] = (170, 120, 170, 120, 170, 120, 175)


class SRSKinematics:

    class Config:
        @staticmethod
        def from_qpos(qpos, arm_angle=0.):
            return SRSKinematics.Config(qpos[1], qpos[3], qpos[5], arm_angle)

        def __init__(self, shoulder, elbow, wrist, arm_angle=0.):
            self.shoulder = np.sign(shoulder)
            self.elbow = np.sign(elbow)
            self.wrist = np.sign(wrist)
            self.arm_angle = wrap_to_pi(arm_angle)

        def __repr__(self):
            return f"Config(shoulder={self.shoulder}, elbow={self.elbow}, wrist={self.wrist}, arm_angle={self.arm_angle:.6f})"

    def __init__(self, srs_params: SRSParams):
        self.params = srs_params
        self.v_0_bs = np.array([0., 0., self.params.d_bs])
        self.v_6_wt = np.array([0., 0., self.params.d_wt])
        self.T_oe_e = np.identity(4) # tcp
        self.T_b_0b = np.identity(4) # user_frame

    def set_tcp(self, tcp) -> None:
        self.T_oe_e = tcp
    
    def get_tcp(self) -> np.ndarray:
        return self.T_oe_e
        
    def set_user_frame(self, user_frame) -> None:
        self.T_b_0b = user_frame
    
    def get_user_frame(self) -> np.ndarray:
        return self.T_b_0b

    def get_fk(self, qpos) -> np.ndarray:
        T = np.eye(4)
        for i, q in enumerate(qpos):
            T = T @ mdh_transform(self.params.mdh[i][0],
                                self.params.mdh[i][1],
                                self.params.mdh[i][2],
                                self.params.mdh[i][3] + q)
        return T

    def get_ik(self, pose, cfg: Config, qpos_seed) -> tuple[KineStatus, np.ndarray]:
        s_psi = np.sin(cfg.arm_angle)
        c_psi = np.cos(cfg.arm_angle)

        p_d = pose[:3,3]
        R_d = pose[:3,:3]
        v_0_sw = p_d - self.v_0_bs - R_d @ self.v_6_wt
        d_sw = np.linalg.norm(v_0_sw)

        # q4
        res_q4, q4 = self._calc_q4(d_sw, cfg.elbow)
        if not res_q4:
            return res_q4, None
        R34 = mdh_transform(self.params.mdh[3][0], 
                            self.params.mdh[3][1], 
                            self.params.mdh[3][2],
                            self.params.mdh[3][3] + q4)[:3,:3]
        # Calc ref q1 q2 q3, when q3 = 0 and arm_angle = 0
        q123_ref = self._calc_ref_q123(v_0_sw, d_sw, cfg.elbow)
        T03_ref = np.eye(4)
        for i, q in enumerate(q123_ref):
            T03_ref = T03_ref @ mdh_transform(self.params.mdh[i][0],
                                            self.params.mdh[i][1],
                                            self.params.mdh[i][2],
                                            self.params.mdh[i][3] + q)
        R03_ref = T03_ref[:3,:3]

        # Calc shoulder q1 q2 q3
        uv_0_sw = v_0_sw / d_sw
        skew_v_0_sw = skew(uv_0_sw)
        As = skew_v_0_sw @ R03_ref
        Bs = - (skew_v_0_sw @ skew_v_0_sw) @ R03_ref
        Cs = np.outer(uv_0_sw, uv_0_sw) @ R03_ref
        R03 = s_psi * As + c_psi * Bs + Cs
        q1, q2, q3 = self._calc_q123(R03, cfg.shoulder, qpos_seed[0])

        # Calc wrist q5 q6 q7
        Aw = R34.T @ As.T @ R_d
        Bw = R34.T @ Bs.T @ R_d
        Cw = R34.T @ Cs.T @ R_d
        R47 = s_psi * Aw + c_psi * Bw + Cw
        q5, q6, q7 = self._calc_q567(R47, cfg.wrist, qpos_seed[4])

        # Check elbow singularity
        if abs(q4) < K_EPS:
            q3 = qpos_seed[2] if qpos_seed is not None else 0.
            q4 = 0.
            T01 = mdh_transform(self.params.mdh[0][0],
                                self.params.mdh[0][1],
                                self.params.mdh[0][2],
                                self.params.mdh[0][3] + q1)
            T12 = mdh_transform(self.params.mdh[1][0],
                                self.params.mdh[1][1],
                                self.params.mdh[1][2],
                                self.params.mdh[1][3] + q2)
            R02 = (T01 @ T12)[:3,:3]
            T56 = mdh_transform(self.params.mdh[5][0],
                                self.params.mdh[5][1],
                                self.params.mdh[5][2],
                                self.params.mdh[5][3] + q6)
            T67 = mdh_transform(self.params.mdh[6][0],
                                self.params.mdh[6][1],
                                self.params.mdh[6][2],
                                self.params.mdh[6][3] + q7)
            R57 = (T56 @ T67)[:3,:3]
            R25 = R02.T @ R_d @ R57.T
            q5 = wrap_to_pi(np.arctan2(R25[2,0], R25[0,0]) - q3)
        # NOTE: 相邻两个同时奇异，或者s和w共线奇异暂时没处理

        qpos = np.array([q1, q2, q3, q4, q5, q6, q7])
        # Check qpos limits
        if not self._check_qps_limits(qpos):
            return KineStatus.OUT_OF_JOINT_LIMITS, qpos
        return KineStatus.OK, qpos

    def calc_arm_angle(self, qpos) -> tuple[KineStatus, float]:
        if not self._check_qps_limits(qpos):
            return KineStatus.OUT_OF_JOINT_LIMITS, None

        if abs(qpos[3]) < K_EPS:
            return KineStatus.OK, 0.
        
        pose = self.get_fk(qpos)
        cfg = SRSKinematics.Config.from_qpos(qpos)

        R_d = pose[:3,:3]
        p_d = pose[:3,3]
        v_26 = p_d - self.v_0_bs - R_d @ self.v_6_wt
        uv_26 = v_26 / np.linalg.norm(v_26)
        v_0_sw = p_d - self.v_0_bs - R_d @ self.v_6_wt
        d_sw = np.linalg.norm(v_0_sw)

        def calc_arm_plane_normal(q1234_, uv_26_):
            T02 = None
            T04 = np.eye(4)
            for i, q in enumerate(q1234_):
                T04 = T04 @ mdh_transform(self.params.mdh[i][0],
                                        self.params.mdh[i][1],
                                        self.params.mdh[i][2],
                                        self.params.mdh[i][3] + q)
                if i == 1:
                    T02 = T04.copy()
            p02 = T02[:3,3]
            p04 = T04[:3,3]
            v_24 = p04 - p02
            uv_24 = v_24 / np.linalg.norm(v_24)
            v_plane = np.cross(uv_24, uv_26_)
            uv_plane = v_plane / np.linalg.norm(v_plane)
            return uv_plane

        # Calc the normal of ref arm angle plane
        q4 = qpos[3]
        q123_ref = self._calc_ref_q123(v_0_sw, d_sw, cfg.elbow)
        uv_ref_plane = calc_arm_plane_normal(np.append(q123_ref, q4), uv_26)

        # Calc the normal of arm angle plane
        uv_plane = calc_arm_plane_normal(qpos[:4], uv_26)

        if abs(np.cross(uv_ref_plane, uv_plane) @ v_26) < K_EPS:
            sign = np.sign(uv_ref_plane[0] * uv_plane[0])
        else:
            sign = np.sign(np.cross(uv_ref_plane, uv_plane) @ v_26)

        psi = sign * safe_acos(uv_ref_plane @ uv_plane)
        return KineStatus.OK, psi

    def calc_feasible_arm_angle_intervals(self, pose, cfg: Config) -> tuple[KineStatus, Intervals]:
        p_d = pose[:3,3]
        R_d = pose[:3,:3]
        v_0_sw = p_d - self.v_0_bs - R_d @ self.v_6_wt
        d_sw = np.linalg.norm(v_0_sw)

        # q4
        res_q4, q4 = self._calc_q4(d_sw, cfg.elbow)
        if not res_q4:
            return res_q4, None
        R34 = mdh_transform(self.params.mdh[3][0], 
                            self.params.mdh[3][1], 
                            self.params.mdh[3][2],
                            self.params.mdh[3][3] + q4)[:3,:3]
        # Calc ref q1 q2 q3, when q3 = 0 and arm_angle = 0
        q123_ref = self._calc_ref_q123(v_0_sw, d_sw, cfg.elbow)
        T03_ref = np.eye(4)
        for i, q in enumerate(q123_ref):
            T03_ref = T03_ref @ mdh_transform(self.params.mdh[i][0],
                                            self.params.mdh[i][1],
                                            self.params.mdh[i][2],
                                            self.params.mdh[i][3] + q)
        R03_ref = T03_ref[:3,:3]

        uv_0_sw = v_0_sw / d_sw
        skew_v_0_sw = skew(uv_0_sw)
        As = skew_v_0_sw @ R03_ref
        Bs = - (skew_v_0_sw @ skew_v_0_sw) @ R03_ref
        Cs = np.outer(uv_0_sw, uv_0_sw) @ R03_ref

        Aw = R34.T @ As.T @ R_d
        Bw = R34.T @ Bs.T @ R_d
        Cw = R34.T @ Cs.T @ R_d

        return KineStatus.OK, self._calc_feasible_arm_angle_intervals(As, Bs, Cs, Aw, Bw, Cw, cfg)


    def get_all_ik(self, pose, cfg: Config, qpos_seed):
        pass

    def get_nearest_ik(self, pose, qpos_seed):
        pass

    def _check_qps_limits(self, qpos) -> bool:
        lower_violate = qpos < (self.params.lb - K_EPS)
        upper_violate = qpos > (self.params.ub + K_EPS)

        if np.any(lower_violate) or np.any(upper_violate):
            idxs = np.where(lower_violate | upper_violate)[0]
            for idx in idxs:
                print(f"q{idx+1} {qpos[idx]} out of limit," + 
                      f"lb:{self.params.lb[idx]}, ub:{self.params.ub[idx]})")
            return False    
        return True        

    def _calc_ref_q123(self, v_0_sw, d_sw, e_cfg: float):
        d_se = self.params.d_se
        d_ew = self.params.d_ew

        # Check shoulder-wrist inline singularity
        if np.linalg.norm(np.cross(v_0_sw, np.array([0.,0.,1.]))) < K_EPS: 
            q1_ref = 0.
        else:
            q1_ref = np.arctan2(v_0_sw[1], v_0_sw[0])

        cos_esw = (d_se**2 + d_sw**2 - d_ew**2) / (2 * d_se * d_sw)
        angle_esw = safe_acos(cos_esw)
        angle_wsz = np.arctan2(np.linalg.norm(v_0_sw[:2]), v_0_sw[2])
        q2_ref = angle_wsz - e_cfg * angle_esw
        q3_ref = 0. # q3 = 0, psi = 0
        # q3_ref = 0. - self.params.mdh[2][3] # q3 = 0, psi = 0
        return np.array([q1_ref, q2_ref, q3_ref])

    def _calc_q123(self, R03, s_cfg: float, q1_seed):
        """R03
        [ *      *   c1s2]       
        [ *      *   s1s2]    
        [-s2c3  s2s3   c2]
        """
        q2 = s_cfg * safe_acos(R03[2,2])
        # Check shoulder sigularity
        if abs(q2) < K_EPS:
            q1 = q1_seed if q1_seed is not None else 0.
            q2 = 0.
            q3 = wrap_to_pi(np.arctan2(R03[1,0], R03[0,0]) - q1)
        else:    
            q1 = np.arctan2(s_cfg * R03[1,2], s_cfg * R03[0,2])
            q3 = np.arctan2(s_cfg * R03[2,1], s_cfg * -R03[2,0])

        return q1, q2, q3

    def _calc_q4(self, d_sw, e_cfg: float) -> tuple[KineStatus, float]:
        d_se = self.params.d_se
        d_ew = self.params.d_ew
        # Calc elbow q4
        if (d_se + d_ew) < d_sw - K_EPS:
            print(f"Pose out of workspace.")
            return KineStatus.OUT_OF_WORKSPACE, None
        cos_q4 = (d_sw**2 - d_se**2 - d_ew**2) / (2 * d_se * d_ew)
        q4 = e_cfg * safe_acos(cos_q4)
        return KineStatus.OK, q4

    def _calc_q567(self, R47, w_cfg: float, q5_seed):
        """R47 
        [ *     *   c5s6]
        [s6c7  s6s7  -c6]
        [ *     *   s5s6]
        """
        q6 = w_cfg * safe_acos(-R47[1,2])
        # Check wrist sigularity
        if abs(q6) < K_EPS:
            q5 = q5_seed if q5_seed is not None else 0.
            q6 = 0.
            q7 = wrap_to_pi(np.arctan2(R47[2,0],R47[0,0]) - q5)
        else:
            q5 = np.arctan2(w_cfg * R47[2,2], w_cfg * R47[0,2])
            # q7 = np.arctan2(w_cfg * R47[1,1], w_cfg * R47[1,0])
            q7 = np.arctan2(w_cfg * -R47[1,1], w_cfg * R47[1,0]) # why -
        
        return q5, q6, q7

    def _calc_feasible_arm_angle_intervals(self, As, Bs, Cs, Aw, Bw, Cw, cfg: Config) -> Intervals:
        q1_psi_intervals = self._calc_tan_feasible_arm_angle_intervals(As[1,2], Bs[1,2], Cs[1,2],
                                                                       As[0,2], Bs[0,2], Cs[0,2],
                                                                       self.params.lb[0], self.params.ub[0], 
                                                                       cfg.shoulder) 
        q2_psi_intervals = self._calc_cos_feasible_arm_angle_intervals(As[2,2], Bs[2,2], Cs[2,2],
                                                                       self.params.lb[1], self.params.ub[1],
                                                                       cfg.shoulder)
        q3_psi_intervals = self._calc_tan_feasible_arm_angle_intervals(As[2,1], Bs[2,1], Cs[2,1],
                                                                       -As[2,0], -Bs[2,0], -Cs[2,0],
                                                                       self.params.lb[2], self.params.ub[2], 
                                                                       cfg.shoulder) 
        q5_psi_intervals = self._calc_tan_feasible_arm_angle_intervals(Aw[2,2], Bw[2,2], Cw[2,2],
                                                                       Aw[2,0], Bw[2,0], Cw[2,0],
                                                                       self.params.lb[4], self.params.ub[4], 
                                                                       cfg.wrist) 
        q6_psi_intervals = self._calc_cos_feasible_arm_angle_intervals(-Aw[1,2], -Bw[1,2], -Cw[1,2],
                                                                       self.params.lb[5], self.params.ub[5],
                                                                       cfg.wrist)
        q7_psi_intervals = self._calc_tan_feasible_arm_angle_intervals(-Aw[1,1], -Bw[1,1], -Cw[1,1],
                                                                       Aw[1,0], Bw[1,0], Cw[1,0],
                                                                       self.params.lb[6], self.params.ub[6], 
                                                                       cfg.wrist) 
        psi_intervals = q1_psi_intervals & q2_psi_intervals & q3_psi_intervals & \
                        q5_psi_intervals & q6_psi_intervals & q7_psi_intervals
        return psi_intervals

    def _calc_cos_joint(self, a, b, c, psi, cfg: float):
        x = a*np.sin(psi) + b*np.cos(psi) + c
        return cfg * safe_acos(x)

    def _calc_tan_joint(self, an, bn, cn, ad, bd, cd, psi, cfg):
        fn = an*np.sin(psi) + bn* np.cos(psi) + cn
        fd = ad*np.sin(psi) + bd* np.cos(psi) + cd
        return np.arctan2(cfg*fn, cfg*fd)

    def _calc_cos_feasible_arm_angle_intervals(self, a, b, c, 
                                               lb, ub, cfg: float) -> Intervals:
        """q2 q6 consin type
        cos(q) = a*sin(psi) + b*cos(psi) + c
               = sqrt(a**2 + b**2) * sin(psi + arctan(b,a)) + c
        """
        # stationary point
        phi = np.arctan2(b, a)
        psi1 = wrap_to_pi(np.pi/2 - phi)
        psi2 = wrap_to_pi(-np.pi/2 - phi)
        q1 = self._calc_cos_joint(a, b, c, psi1, cfg)
        q2 = self._calc_cos_joint(a, b, c, psi2, cfg)
        q_min = min(q1, q2)
        q_max = max(q1, q2)
        lb_ = max(0., lb) if cfg > 0. else min(0., lb)
        ub_ = max(0., ub) if cfg > 0. else min(0., ub)
        interval1 = Interval(q_min, q_max)
        interval2 = Interval(lb_, ub_)
        if interval1.intersect(interval2).is_empty():
            return Intervals.empty()
        if interval2.contains_interval(interval1):
            return Intervals([Interval(-np.pi, np.pi)])

        split_points = [-np.pi, np.pi]
        res1 = solve_sin_cos_eq(a, b, c-np.cos(lb_))
        if res1[0] == SolutionStatus.FINITE:
            split_points.extend(res1[1])
        res2 = solve_sin_cos_eq(a, b, c-np.cos(ub_))
        if res2[0] == SolutionStatus.FINITE:
            split_points.extend(res2[1])

        def verif_func(self, psi):
            q = self._calc_cos_joint(a, b, c, psi, cfg)
            return lb - K_EPS <= q <= ub + K_EPS

        return intervals_from_split_points(split_points, verif_func)

    def _calc_tan_feasible_arm_angle_intervals(self, an, bn, cn, 
                                               ad, bd, cd, 
                                               lb, ub, cfg: float) -> Intervals:
        split_points = [-np.pi, np.pi]

        at = cfg * (bd*cn - bn*cd)
        bt = cfg * (an*cd - ad*cn)
        ct = cfg * (an*bd - ad*bn)
        cond = at*at + bt*bt - ct*ct
        if cond > K_EPS: # cyclic
            # stationary point
            psi1 = wrap_to_pi(2*np.arctan((at + safe_sqrt(cond))/(bt-ct))) 
            psi2 = wrap_to_pi(2*np.arctan((at - safe_sqrt(cond))/(bt-ct)))
            q1 = self._calc_tan_joint(an, bn, cn, ad, bd, cd, psi1, cfg)
            q2 = self._calc_tan_joint(an, bn, cn, ad, bd, cd, psi2, cfg)
            q_min = min(q1, q2)
            q_max = max(q1, q2)
            interval1 = Interval(q_min, q_max)
            interval2 = Interval(lb, ub)
            if interval1.intersect(interval2).is_empty():
                return Intervals.empty()

        def best_psi(q_target, psi_candidates):
            if len(psi_candidates) < 2:
                return psi_candidates
            return min(psi_candidates, 
                       key=lambda psi: abs(self._calc_tan_joint(an, bn, cn, ad, bd, cd, psi, cfg) - q_target))

        res1 = solve_sin_cos_eq(an-ad*np.tan(lb), bn-bd*np.tan(lb), cn-cd*np.tan(lb))
        if res1[0] == SolutionStatus.FINITE:
            if cond < -K_EPS: # Monotonic
                split_points.append(best_psi(lb, res1[1]))
            else:
                split_points.extend(res1[1])
        res2 = solve_sin_cos_eq(an-ad*np.tan(ub), bn-bd*np.tan(ub), cn-cd*np.tan(ub))
        if res2[0] == SolutionStatus.FINITE:
            if cond < -K_EPS: # Monotonic
                split_points.append(best_psi(ub, res2[1]))
            else:
                split_points.extend(res2[1])

        def verif_func(psi):
            q = self._calc_tan_joint(an, bn, cn, ad, bd, cd, psi, cfg)
            return lb - K_EPS <= q <= ub + K_EPS

        return intervals_from_split_points(split_points, verif_func)


def intervals_from_split_points(split_points: list[float], func):
    sorted_points = sorted(split_points)
    unique_points: list[float] = []
    for point in sorted_points:
        if not unique_points or abs(point - unique_points[-1]) > K_EPS_SMALL:
            unique_points.append(point)

    if len(unique_points) < 2:
        return Intervals.empty()

    intervals: list[Interval] = []
    lb = unique_points[0]
    for ub in unique_points[1:]:
        mid = 0.5 * (lb + ub)
        if func(mid):
            intervals.append(Interval(lb, ub))
        lb = ub

    return Intervals(intervals)
