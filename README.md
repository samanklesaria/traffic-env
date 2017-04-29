Code Overview
=============
- The 'args.py' file handles argument parsing and the creation of derived flags. It runs any functions that derive flags from other flags repeatedly until it finds a fixed point. Default arguments for reinforcement learning are defined in 'alg\_flags.py'
- The 'algorithms' directory contains reinforcement learning algorithms. Currently I support asynhronous advantage actor critic, double dueling q learning with and without recurrent units, vanilla policy gradient, and the cross entropy method. For comparison, 'random' makes random choices, 'const0' always chooses action 0, 'const1' chooses action 1, 'fixed' cycles between the two, waiting the number of steps given by the SPACING flag, and 'greedy' chooses the direction with more waiting cars.
- The file 'util.py' in the algorithms directory contains common patterns in reinforcement learning: e-greedy strategies, boltzman sampling, running an episode, discounting rewards, etc. 
- The simulation environment is defined in the file 'envs/traffic env.py'. It obeys the Intelligent Driver Model. The grid topology is defined in 'envs/roadgraph.py'. 
- Because gym is designed with single-agent environments in mind, replaced the gym's notion of a space with what I called a GSpace (generic space). This code is in 'spaces/gspace.py'. There are other monkey patches in the main init file that allow environments to render at a different timescale than they interactor with actors. 

Reproducing Training Sessions
=============================
The settings used for each session are saved to a settings.json file in the log directory. This means you can change default params for an algorithm without sacrificing the reproducibility of previous sessions.
The tensorflow graph used in a run is also saved to disk. This means that as long as the input and output nodes don't change, you an play with the neural net architecture without breaking previous runs.

Intelligent Driver Model Implementation
=============================
I use a ring buffer to store the cars on each road. The IDM requires each
car to have a leading car, so we introduce a fake car that the front car can follow. 
On red lights, this fake car has position at the end of the road. On green lights, it's at
the position of the first car on the next road ahead, or +inf if no such car exists.
The index of this fake car is leading[rd]. The car with no followers is stored in lastcar[rd].
If lastcar[rd] == leading[rd], then the buffer is empty. When a car leaves the road, that
car's spot becomes leading[rd]. When a car is added, that car's spot becomes lastcar[rd].
We reserve index 0 as a copy of index -1, so sim can work with consecutive array entries.
