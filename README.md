# ActiveInferenceBall
Minimal example where an active inference-driven assistant helps a ball to move as far as possible.

There are two versions of the model:
* `velocity_only`: We assume that the velocity transition probabilities are conditional only on 
  the current velocity and the context
(here the time).
* `velocity_and_position`: We assume that the velocity transition probabilities are conditional 
  on the current velocity, the current position, and the context

### The 'true' generative model

The 'true' generative model is a Gaussian Process with a squared exponential kernel.

We assume that the force of the ball is a Gaussian process with a squared exponential kernel. 
The kernel is defined a regular squared exponential kernel:
$$
k(x, x') = \alpha^2 \exp\left(-\frac{1}{2l^2} (x - x')^2\right)
$$
where $\alpha$ is the amplitude and $l$ is the length scale. 
The mean of the Gaussian process is a cosine function. 
Note that all of this is completely arbitrary.


### ***Note to Self: Debugging***

* When changing the env, make sure to check that the mesh of grid is not too tight or too 
  ('oversampling').
* When changing the env, make sure to check that the distribution used for the preferences gives 
  enough contrast between the different options.
* Two possible consequences of not doing what is mentionned above is that:
  - a low factor in front of the 
    epistemic value will work better than a high factor;
  - simply not beating the baseline at all.


### ***Note to Self: Possible Extensions***

- Always use a fix horizon (modulo the beginning of the episode)
- Pretraining the model
- Using a couple of different generative models
- Add meaningful priors (e.g. 'day' vs 'night')
- Add the current position as a predictor of the velocity