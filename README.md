# learning-policy-gradients

PPO from scratch (PyTorch + Gymnasium) on continuous-control `LunarLander-v3`.
Clipped surrogate objective, GAE(λ), a tanh-squashed Gaussian policy with the
log-prob correction, and separate actor/critic networks with learning-rate and
entropy schedules.

## Demo

https://github.com/Maanav-s/learning-policy-gradient/raw/Lunar_Lander/videos/lunarlander_demo_2ep.mp4

Two consecutive evaluation episodes (seeds 40 and 15, returns +286.5 and +278.1)
from the policy in `checkpoints/policy.pt`.

## Results

100 deterministic evaluation episodes, seeds 0–99:

| metric | value |
| --- | --- |
| mean return | +240.6 ± 6.6 (SEM) |
| median return | +253.7 |
| p5 / p95 | +40.3 / +292.7 |
| return ≥ 200 | 92% (95% CI 85.0–95.9%) |
| mean episode length | 223 steps |

## Usage

```powershell
uv run python main.py train                      # train, writes checkpoints/ and figures/
uv run python main.py eval --episodes 100        # headless evaluation statistics
uv run python main.py demo --checkpoint checkpoints/policy.pt   # watch it live
uv run python main.py record --episodes 20 --keep 5 --gif       # save mp4/gif clips
```

`record` rolls out candidate episodes offscreen and keeps only the best ones by
return, as web-ready H.264 mp4 (and optionally gif). Note that return alone is a
poor proxy for how a landing *looks*: the highest-scoring episodes often touch
down well off the pad and then slide into it, so the committed demo was chosen by
touchdown offset and post-contact drift instead.
