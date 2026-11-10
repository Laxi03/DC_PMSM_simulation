"""Shared TD3 training and explicit checkpoint provenance."""
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

from .experiments import Experiment, environment_factory


def train(case: Experiment, output: Path, *, steps: int, workers: int, device: str = "cpu") -> Path:
    import numpy as np
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "td3-drive-control/training-v1",
        "experiment": asdict(case),
        "requested_steps": steps,
        "workers": workers,
        "device": device,
        "status": "running",
        "versions": {name: version(name) for name in ("stable-baselines3", "torch", "gymnasium")},
    }

    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    env = None
    try:
        factories = [environment_factory(case, training=True, rank=i) for i in range(workers)]
        env = DummyVecEnv(factories) if workers == 1 else SubprocVecEnv(factories, start_method="spawn")
        dimensions = env.action_space.shape[0]
        noise = NormalActionNoise(mean=np.zeros(dimensions), sigma=np.full(dimensions, case.noise_std))
        model = TD3(
            "MlpPolicy", env, verbose=1, learning_rate=3e-4, buffer_size=200_000,
            batch_size=128, gamma=0.99, tau=0.005, train_freq=(1, "step"),
            gradient_steps=1, action_noise=noise, policy_kwargs={"net_arch": [128, 64]},
            seed=case.seed, device=device,
        )
        model.learn(total_timesteps=steps)
        checkpoint = output / "model.zip"
        model.save(checkpoint)
        manifest.update(status="complete", completed_steps=model.num_timesteps)
        return checkpoint
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        if env is not None:
            env.close()
        save_manifest()
