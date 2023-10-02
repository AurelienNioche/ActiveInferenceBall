# ActiveInferenceBall
Minimal example where an active inference-driven assistant helps a ball to move as far as possible.

There are two versions of the model:
* `velocity_only`: We assume that the velocity transition probabilities are conditional only on 
  the current velocity and the context
(here the time).
* `velocity_and_position`: We assume that the velocity transition probabilities are conditional 
  on the current velocity, the current position, and the context


### ***Note to Self: Possible Extensions***

- Always use a fix horizon (modulo the beginning of the episode)
- Pretraining the model
- Using a couple of different generative models
- Add meaningful priors (e.g. 'day' vs 'night')
- Add the current position as a predictor of the velocity