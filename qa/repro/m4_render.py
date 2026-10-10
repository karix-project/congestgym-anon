"""M4: render() returns None although rgb_array is an advertised render mode."""
import matplotlib

matplotlib.use("Agg")

from carla_gym.env import CarlaGymEnv

for mode in ("rgb_array", "human"):
    env = CarlaGymEnv(render_mode=mode, max_steps=5, seed=0)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    out = env.render()
    shape = getattr(out, "shape", None)
    print(f"render_mode={mode!r}: returned {type(out).__name__}, shape={shape}")
    env.close()
