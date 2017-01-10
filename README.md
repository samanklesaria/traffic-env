What's Included
===============
- A fast, thread safe gym environment called TrafficEnv for simulating the intelligent
driver model on any network topology. Written using numba for performance.
- A network topology for grids
- A HistoryWrapper, which is like gym's SkipWrapper, but instead
of discarding intervening observations, it samples them at regular 
intervals and returns an array of the sampled observations.
- A cross entropy method trainer which works on multidimensional
action and reward spaces.
- A simply policy gradient trainer for multidimensional spaces.
- An asynchronous advantage actor critic trainer for multidimensional spaces.

Potential Points of Confusion
=============================

All trainers have the same interface: they define a 'run' function
that takes a function that returns an environment to train. They
take a function, not an environment, because trainers like A3C
need to be able to make multiple environments.

All trainers support saving their weights to a tensorflow checkpoint
directory. They also support validating previously
trained policies with the 'validate' function.

The gym Env class has been monkey patched so that 'step' will call
'render' if the environment's 'rendering' attribute is true. This
is necessary if you want to render each frame of a wrapped environment.
This way, if you want to render and validate a policy at the same time,
just pass an environment with the 'rendering' attribute turned on to
a trainer's 'validate' function.

The FLAGS object from tensorflow.app is abused to allow each
component to specify its command line arguments.
This is easier than centralizing all command line arguments
and passing them each component in its constructor, which is
the standard approach. 
