from utils import (
    TurningController,
    left_turns_descending,
    right_turns_descending,
    SimpleHeadStabilisedFly,
)
from flygym import YawOnlyCamera
from flygym.arena import FlatTerrain
from flygym.examples.path_integration.controller import (
    RandomExplorationController,
)

from tqdm import trange
import numpy as np
import pickle

def run_exploration(seed, run_time, output_dir):

    arena = FlatTerrain()

    fly = SimpleHeadStabilisedFly()

    cam_params = {"mode": "track", "pos": (0, 0, 30), "euler": (0, 0, 0), "fovy": 60}
    cam = YawOnlyCamera(
        attachment_point=fly.model.worldbody,
        targeted_fly_names=[fly.name],
        camera_name="birdeye_cam",
        timestamp_text=False,
        camera_parameters=cam_params,
    )

    sim = TurningController(
        fly=fly, cameras=[cam],
        arena=arena, timestep=1e-4,
        intrinsic_freqs=np.ones(6)*25
        )


    random_exploration_controller = RandomExplorationController(
        dt=sim.timestep,
        lambda_turn=2,
        seed=seed,
        forward_dn_drive=(1.0, 1.0),
        left_turn_dn_drive=left_turns_descending,
        right_turn_dn_drive=right_turns_descending,
    )

    obs, info = sim.reset(0)
    obs_hist, info_hist, action_hist = [], [], []

    for i in trange(int(run_time / sim.timestep)):
        walking_state, dn_drive = random_exploration_controller.step()
        action_hist.append(dn_drive)
        obs, reward, terminated, truncated, info = sim.step(dn_drive)

        im = sim.render()

        obs_hist.append(obs)
        info_hist.append(info)

    # Save data if output_dir is provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cam.save_video(output_dir / "rendering.mp4")
        with open(output_dir / "sim_data.pkl", "wb") as f:
            data = {
                "obs_hist": obs_hist,
                "info_hist": info_hist,
                "action_hist": action_hist,
            }
            pickle.dump(data, f)