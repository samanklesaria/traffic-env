What's Included
===============
- A reasonably fast and thread safe gym environment called TrafficEnv. It can simulate the intelligent
driver model on any network topology. Written using numba for performance.
- A network topology for grids.
- A StrobeWrapper, which is like gym's SkipWrapper, but instead
of discarding intervening observations, it samples them at regular 
intervals and returns an array of the sampled observations.
- A HistoryWrapper, which bundles up the observations of previous steps into an array.
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

For the TrafficEnv class, we use a ring buffer to store the cars on each road. The IDM requires each
car to have a leading car, so we introduce a fake car that the front car can follow. 
On red lights, this fake car has position at the end of the road. On green lights, it's at
the position of the first car on the next road ahead, or +inf if no such car exists.
The index of this fake car is leading[rd]. The car with no followers is stored in lastcar[rd].
If lastcar[rd] == leading[rd], then the buffer is empty. When a car leaves the road, that
car's spot becomes leading[rd]. When a car is added, that car's spot becomes lastcar[rd].
We reserve index 0 as a copy of index -1, so sim can work with consecutive array entries.
