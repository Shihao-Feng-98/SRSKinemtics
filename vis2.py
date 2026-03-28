from IPython import embed
import numpy as np
from scipy.spatial.transform import Rotation as R
import pathlib
from loop_rate_limiters import RateLimiter
from trimesh.scene import Scene
import viser
from viser.extras import ViserUrdf

import time
from src.srs_kinematics import SRSKinematics, KineStatus, iiwa14


def rotation_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = R.from_matrix(rotation).as_quat()
    return np.roll(xyzw, 1)

def wxyz_to_rotation(wxyz) -> np.ndarray:
    xyzw = np.roll(wxyz, -1)
    return R.from_quat(xyzw).as_matrix()

def pos_wxyz_to_homogeneous(pos: np.ndarray,
                            wxyz: np.ndarray = [1, 0, 0, 0]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = wxyz_to_rotation(wxyz)
    return T

def viser_name_from_frame(
    scene: Scene,
    frame_name: str,
    root_node_name: str = "/",
) -> str:
    """Given the (unique) name of a frame in our URDF's kinematic tree, return a
    scene node name for viser.

    For a robot manipulator with four frames, that looks like:


            ((shoulder)) == ((elbow))
               / /             |X|
              / /           ((wrist))
         ____/ /____           |X|
        [           ]       [=======]
        [ base_link ]        []   []
        [___________]


    this would map a name like "elbow" to "base_link/shoulder/elbow".
    """
    assert root_node_name.startswith("/")
    assert len(root_node_name) == 1 or not root_node_name.endswith("/")

    frames = []
    while frame_name != scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = scene.graph.transforms.parents[frame_name]
    if root_node_name != "/":
        frames.append(root_node_name)
    return "/".join(frames[::-1])



if __name__ == "__main__":
    vis = viser.ViserServer()
    vis.scene.world_axes.visible = True
    h_grid = vis.scene.add_grid(name="/ground",
                                        width=10,
                                        height=10)
    # robot 1
    frame_base_1 = "/robot_base_1"
    h_base_1 = vis.scene.add_frame(name=frame_base_1, 
                                    show_axes=True,
                                    axes_length=0.2,
                                    axes_radius=0.005,
                                    position=(0,0,1), 
                                    wxyz=rotation_to_wxyz(np.array([[0,-1,0],[0,0,1],[-1,0,0]])))
    h_robot_1 = ViserUrdf(vis,
                            urdf_or_path=pathlib.Path("./assets/iiwa_description/iiwa14_spheres_dense_collision.urdf"),
                            root_node_name=frame_base_1,
                            load_meshes=True,
                            load_collision_meshes=False)
    h_robot_1.update_cfg(np.deg2rad([0, 75, -90, 75, 0, 30, 0]))
    frame_ee_1 = f'{frame_base_1}/visual/{viser_name_from_frame(h_robot_1._urdf.scene, "iiwa_link_ee_kuka")}/frame'
    h_ee_1 = vis.scene.add_frame(frame_ee_1,
                                show_axes=True,
                                axes_length=0.1,
                                axes_radius=0.005,
                                position=(0, 0, 0),
                                wxyz=(1, 0, 0, 0))
    
    # 身体圆柱
    sphere_positions = [(0,0,0.2),(0,0,0.4),(0,0,0.6),(0,0,0.8)]
    sphere_colors = [[0,255,0]]*len(sphere_positions)
    for i, (pos, color) in enumerate(zip(sphere_positions, sphere_colors)):
        h_sphere = vis.scene.add_icosphere(name=f"/robot_1/sphere_{i}", radius=0.25, position=pos, color=color, opacity=0.5)
    # 肘部球
    elbow_spere_1 = f'{frame_base_1}/visual/{viser_name_from_frame(h_robot_1._urdf.scene, "iiwa_link_4")}/sphere'
    vis.scene.add_icosphere(name=elbow_spere_1, radius=0.12, position=(0,0,0), color=(255,0,0), opacity=0.5)

    kine = SRSKinematics(iiwa14())
    kine.set_user_frame(pos_wxyz_to_homogeneous(h_base_1.position,
                                                h_base_1.wxyz))

    vis.gui.configure_theme(control_width="large")
    # 状态显示
    h_qpos_text_1 = vis.gui.add_text(label="qpos_1(rad)", initial_value="[]")
    h_arm_angle_text_1 = vis.gui.add_text(label="arm_angle_1(rad)", initial_value="[]")
    h_ee_pose_text_1 = vis.gui.add_text(label="ee_pose_1", initial_value="[]")

    cur_pose_1 = kine.get_fk(h_robot_1._urdf.cfg.copy())
    h_ee_goal_1 = vis.scene.add_transform_controls(name="/ee_target_1",
                                                    scale=0.2,
                                                    position=cur_pose_1[:3, 3],
                                                    wxyz=rotation_to_wxyz(cur_pose_1[:3, :3]))

    # robot 2
    frame_base_2 = "/robot_base_2"
    h_base_2 = vis.scene.add_frame(name=frame_base_2, 
                                    show_axes=True,
                                    axes_length=0.2,
                                    axes_radius=0.005,
                                    position=(0,1.2,1), 
                                    wxyz=rotation_to_wxyz(np.array([[0,-1,0],[0,0,1],[-1,0,0]])))
    h_robot_2 = ViserUrdf(vis,
                            urdf_or_path=pathlib.Path("./assets/iiwa_description/iiwa14_spheres_dense_collision.urdf"),
                            root_node_name=frame_base_2,
                            load_meshes=True,
                            load_collision_meshes=False)
    h_robot_2.update_cfg(np.deg2rad([0, 75, -90, 75, 0, 30, 0]))
    frame_ee_2 = f'{frame_base_2}/visual/{viser_name_from_frame(h_robot_2._urdf.scene, "iiwa_link_ee_kuka")}/frame'
    h_ee_2 = vis.scene.add_frame(frame_ee_2,
                                show_axes=True,
                                axes_length=0.1,
                                axes_radius=0.005,
                                position=(0, 0, 0),
                                wxyz=(1, 0, 0, 0))

    # 身体圆柱
    sphere_positions = [(0,1.2,0.2),(0,1.2,0.4),(0,1.2,0.6),(0,1.2,0.8)]
    sphere_colors = [[0,255,0]]*len(sphere_positions)
    for i, (pos, color) in enumerate(zip(sphere_positions, sphere_colors)):
        h_sphere = vis.scene.add_icosphere(name=f"/robot_2/sphere_{i}", radius=0.25, position=pos, color=color, opacity=0.5)
    # 肘部球
    elbow_spere_2 = f'{frame_base_2}/visual/{viser_name_from_frame(h_robot_2._urdf.scene, "iiwa_link_4")}/sphere'
    vis.scene.add_icosphere(name=elbow_spere_2, radius=0.12, position=(0,0,0), color=(255,0,0), opacity=0.5)

    # 状态显示
    h_qpos_text_2 = vis.gui.add_text(label="qpos_2(rad)", initial_value="[]")
    h_arm_angle_text_2 = vis.gui.add_text(label="arm_angle_2(rad)", initial_value="[]")
    h_ee_pose_text_2 = vis.gui.add_text(label="ee_pose_2", initial_value="[]")


    # embed(banner1="Start")
    # 实时跟踪
    rate = RateLimiter(frequency=50, warn=True)
    t0 = time.time()
    pre_goal = pos_wxyz_to_homogeneous(h_ee_goal_1.position, 
                                        h_ee_goal_1.wxyz)
    while time.time() - t0 < 60:
        # robot 1
        cur_qpos_1 = h_robot_1._urdf.cfg.copy()
        h_qpos_text_1.value = f"{np.round(cur_qpos_1, 3).tolist()}"

        cur_pose_1 = kine.get_fk(cur_qpos_1)
        h_ee_pose_text_1.value = f"{np.round(cur_pose_1[:3,3], 3).tolist()}," + \
                                    f"{np.round(rotation_to_wxyz(cur_pose_1[:3,:3]), 3).tolist()}"
        
        cur_psi_1 = kine.calc_arm_angle(cur_qpos_1)
        cfg_1 = SRSKinematics.Config.from_qpos(cur_qpos_1, cur_psi_1)
        res, intervals_1 = kine.calc_feasible_arm_angle_intervals(cur_pose_1, cfg_1)
        if res == KineStatus.OK:
            for interval in intervals_1:
                if interval.contains(cur_psi_1):
                    h_arm_angle_text_1.value = f"{np.round(cur_psi_1, 3)} ∈{interval}"
        else:
            h_arm_angle_text_1.value = "Error"

        # robot 2
        cur_qpos_2 = h_robot_2._urdf.cfg.copy()
        h_qpos_text_2.value = f"{np.round(cur_qpos_2, 3).tolist()}"

        cur_pose_2 = kine.get_fk(cur_qpos_2)
        h_ee_pose_text_2.value = f"{np.round(cur_pose_2[:3,3], 3).tolist()}," + \
                                    f"{np.round(rotation_to_wxyz(cur_pose_2[:3,:3]), 3).tolist()}"
        
        cur_psi_2 = kine.calc_arm_angle(cur_qpos_2)
        cfg_2 = SRSKinematics.Config.from_qpos(cur_qpos_2, cur_psi_2)
        res, intervals_2 = kine.calc_feasible_arm_angle_intervals(cur_pose_2, cfg_2)
        if res == KineStatus.OK:
            for interval in intervals_2:
                if interval.contains(cur_psi_2):
                    h_arm_angle_text_2.value = f"{np.round(cur_psi_2, 3)} ∈{interval}"
        else:
            h_arm_angle_text_2.value = "Error"



        goal_pose_1 = pos_wxyz_to_homogeneous(h_ee_goal_1.position, 
                                                h_ee_goal_1.wxyz)
        if np.allclose(goal_pose_1, pre_goal, atol=1e-4):
            rate.sleep()
            continue
        pre_goal = goal_pose_1
        # robot 1
        res, goal_qpos_1 = kine.get_next_ik(goal_pose_1, cur_qpos_1, False)
        if res == KineStatus.OK:
            if not np.allclose(goal_qpos_1, cur_qpos_1, atol=np.deg2rad(90)):
                print("Warning: IK solution discontinuous!" + \
                      f"cur_qpos_1={np.round(cur_qpos_1, 3)}, goal_qpos_1={np.round(goal_qpos_1, 3)}")
            h_robot_1.update_cfg(goal_qpos_1)
        # robot 2
        res, goal_qpos_2 = kine.get_nearest_ik(goal_pose_1, cur_qpos_2, False)
        if res == KineStatus.OK:
            if not np.allclose(goal_qpos_2, cur_qpos_2, atol=np.deg2rad(90)):
                print("Warning: IK solution discontinuous!" + \
                      f"cur_qpos_2={np.round(cur_qpos_2, 3)}, goal_qpos_2={np.round(goal_qpos_2, 3)}")
            h_robot_2.update_cfg(goal_qpos_2)

        rate.sleep()  

    # 绘制曲线图