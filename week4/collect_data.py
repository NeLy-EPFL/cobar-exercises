# This script collects training data for 2_neural_network.ipynb.
# The outputs have already been saved to ./data/data.npz as it
# may take a while to run.
import numpy as np
from flygym.arena import FlatTerrain
from flygym import Fly

class ArenaWithFly2(FlatTerrain):
    def __init__(self, height=0.2, **kwargs):
        super().__init__(**kwargs)

        self.height = height
        fly = Fly().model
        fly.model = "Animat_2"

        for light in fly.find_all(namespace="light"):
            light.remove()

        spawn_site = self.root_element.worldbody.add(
            "site",
            pos=(0, 0, self.height),
            euler=(0, 0, 0),
        )
        self.freejoint = spawn_site.attach(fly).add("freejoint")

    def set_fly2_position(self, physics, obs, r, theta, phi):
        fly1_heading = obs["fly_orientation"][:2] @ (1, 1j)
        fly1_heading /= np.abs(fly1_heading)
        fly1_pos = obs["fly"][0, :2] @ (1, 1j)

        fly2_pos = r * np.exp(1j * theta) * fly1_heading + fly1_pos
        fly2_heading = np.exp(1j * phi) * fly1_heading

        q = np.exp(1j * np.angle(fly2_heading) / 2)
        qpos = (fly2_pos.real, fly2_pos.imag, self.height, q.real, 0, 0, q.imag)
        physics.bind(self.freejoint).qpos = qpos


if __name__ == "__main__":
    from tqdm import trange
    from flygym.examples.locomotion import HybridTurningController
    from utils import crop_hex_to_rect
    from pathlib import Path

    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    arena = ArenaWithFly2()

    fly = Fly(
        enable_vision=True,
        init_pose="tripod",
        contact_sensor_placements=contact_sensor_placements,
        vision_refresh_rate=np.inf,
    )

    sim = HybridTurningController(
        fly=fly,
        timestep=1e-4,
        arena=arena,
    )

    obs, info = sim.reset(seed=0)
    images = []

    n_steps = 10000
    rng = np.random.RandomState(0)

    r = rng.uniform(1.5, 10, n_steps)
    theta = rng.uniform(-np.pi, np.pi, n_steps)
    phi = rng.uniform(-np.pi, np.pi, n_steps)

    for i in trange(500):
        obs = sim.step(np.array([1, 1]))[0]

    for i in trange(n_steps):
        obs = sim.step(np.array([1, 1]))[0]
        arena.set_fly2_position(sim.physics, obs, r=r[i], theta=theta[i], phi=phi[i])
        obs = sim.get_observation()
        im_fly = crop_hex_to_rect(obs["vision"])
        images.append(im_fly)

    images = np.array(images, dtype=np.float32)
    r = r.astype(np.float32)
    theta = theta.astype(np.float32)
    Path("data").mkdir(exist_ok=True)
    np.savez_compressed("data/data.npz", images=images, r=r, theta=theta)
