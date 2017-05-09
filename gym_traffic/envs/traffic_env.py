import gym
from gym_traffic.spaces.gspace import GSpace
import numpy as np
from numba import jit, void, float64, float32, int64, int32, uint32, boolean
import numba
import itertools
import time
import math
from args import FLAGS, add_argument

add_argument('--local_cars_per_sec', 0.12, type=float)
add_argument('--rate', 0.5, type=float)
add_argument('--poisson',True, type=bool)
add_argument('--entry', 'all')
add_argument('--learn_switch', False, type=bool)

THRESH = 0.1

# Python attribute access is expensive. We hardcode these params
PASSING_REWARD = 0
YELLOW_TICKS = 6
DECEL_PENALTY = False
OVERFLOW_PENALTY = 10
CAPACITY = 20
EPS = 1e-8

# Get the rotation of a line segment
def get_rot(line, length):
  l = line / length
  return np.arctan2(l[1,1] - l[0,1], l[1,0] - l[0,0])

# New cars have parameters sampled uniformly from archetypes
params = 10
xi, vi, li, ai, deltai, v0i, bi, ti, s0i, wi = range(params)
archetypes = np.zeros((1, params), dtype=np.float32)
archetypes[0,vi] = 11.11
archetypes[0,ai] = 3
archetypes[0,deltai] = 4
archetypes[0,v0i] = 13.89
archetypes[0,li] = 4
archetypes[0,bi] = 6
archetypes[0,ti] = 2
archetypes[0,s0i] = 1

# Like mod, but preserves index 0
@jit(int32(int32), nopython=True,nogil=True,cache=True)
def wrap(a): return 1 if a >= CAPACITY else a

# The Intelligent Driver Model of forward acceleration
@jit(float32[:](float32,float32[:,:],float32[:,:]), nopython=True,nogil=True,cache=True)
def sim(r, ld, me):
  v = me[vi]
  s_star = me[s0i] + np.maximum(0, v*me[ti] + v *
          (v - ld[vi]) / (2 * np.sqrt(me[ai]*me[bi])))
  s = ld[xi] - me[xi] - ld[li]
  dv = (me[ai] * (1 - (v / me[v0i])**me[deltai] -
    np.square(s_star / (s + EPS)))).astype(np.float32)
  dvr = dv*r
  dx = r*v + 0.5*dvr*r
  me[xi] += (dx > 0)*dx
  me[vi] = np.maximum(0, v + dvr)
  return dvr

@jit(void(int32[:],int32[:],int32[:],float32[:],
  boolean[:],int32[:]), nopython=True,nogil=True,cache=True)
def remi(dests,phases,current_phase,rewards,passed_dst,waiting):
  # if cars are stopped on red roads while no passing occurs, -= 1
  # if passing occurs and no cars are stopped on green roads += 1
  rewards[:] = 0
  for (e,dst) in enumerate(dests):
    if dst == -1: break
    green = phases[e] != current_phase[dst]
    if waiting[e] > 0 and not green and not passed_dst[dst]:
      rewards[dst] -= 0.5
    elif passed_dst[dst] and green and not (waiting[e] > 0):
      rewards[dst] += 0.5
  passed_dst[:] = False

# Update the leading car at the end of each road depending on light phases
@jit(void(int32[:],int32[:],float32,int32[:],float32[:,:,:],int32[:],int32[:],int32[:],int32[:]),
    nopython=True,nogil=True,cache=True)
def update_lights(dests,phases,length,nexts,state,leading,lastcar,current_phase,elapsed):
  for (e,dst) in enumerate(dests):
    if dst == -1: return
    if phases[e] == current_phase[dst] or elapsed[dst] < YELLOW_TICKS:
      state[e, xi, leading[e]] = length
    else:
      newrd = nexts[e]
      if newrd >= 0 and lastcar[newrd] != leading[newrd]:
        state[e, xi, leading[e]] = state[newrd, xi, lastcar[newrd]]
        state[e, xi, leading[e]] += length
      else:
        state[e, xi, leading[e]] = np.inf

# Add a new car to a road
@jit(boolean(int32,float32[:],float32[:,:,:],int32[:],int32[:],
  float32[:],int32[:]),nopython=True,nogil=True,cache=True)
def add_car(road,car,state,leading,lastcar,rewards,dests):
  pos = wrap(lastcar[road] + 1)
  start_pos = np.inf
  if lastcar[road] != leading[road]:
    start_pos = state[road,xi,lastcar[road]] - state[road,li,lastcar[road]] \
        - state[road,s0i,lastcar[road]]
  if pos != leading[road]:
    state[road,:,pos] = car
    state[road,xi,pos] = min(state[road,xi,pos], start_pos)
    lastcar[road] = pos
  elif dests[road] >= 0:
    rewards[dests[road]] -= OVERFLOW_PENALTY
    return True
  else:
    return True
  return False

# Remove cars with x coordinates beyond their roads' lengths
@jit(boolean(int32[:],float32,int32[:],float32[:,:,:],int32[:],int32[:],int32[:],float32[:],
  boolean[:]),nopython=True,nogil=True,cache=True)
def advance_finished_cars(dests,length,nexts,state,leading,lastcar,passed,
    rewards,passed_dst):
  overflowed = False
  for e in range(nexts.shape[0]):
    while leading[e] != lastcar[e] and state[e,xi,wrap(leading[e]+1)] > length:
      newlead = wrap(leading[e]+1)
      newrd = nexts[e]
      if newrd >= 0:
        passed[e] += 1
        passed_dst[dests[e]] = True
        rewards[dests[e]] += PASSING_REWARD
        state[e,xi,newlead] -= length
        overflowed = add_car(newrd,state[e,:,newlead],state,
            leading,lastcar,rewards,dests) or overflowed
      state[e,:,newlead] = state[e,:,leading[e]]
      leading[e] = newlead
  return overflowed

# Remove cars with x coordinates beyond their roads' lengths, adding these cars'
# travel times to a list if they go off the map
def advance_hack(dests,length,nexts,state,leading,lastcar,passed,
    rewards,passed_dst, trip_times, tick):
  overflowed = False
  for e in range(nexts.shape[0]):
    while leading[e] != lastcar[e] and state[e,xi,wrap(leading[e]+1)] > length:
      newlead = wrap(leading[e]+1)
      newrd = nexts[e]
      if newrd >= 0:
        passed[e] += 1
        passed_dst[dests[e]] = True
        rewards[dests[e]] += PASSING_REWARD
        state[e,xi,newlead] -= length
        overflowed = add_car(newrd,state[e,:,newlead],state,
            leading,lastcar,rewards,dests) or overflowed
      else:
        trip_times.append((tick - state[e,wi,newlead]) / 2)
      state[e,:,newlead] = state[e,:,leading[e]]
      leading[e] = newlead
  return overflowed

# Yields None separated groups of incoming cars for each tick according to Poisson
def poisson(random):
  lam = 1 / (FLAGS.cars_per_sec * FLAGS.rate)
  while True:
    for _ in range(round(random.exponential(lam))): yield None
    yield archetypes[random.randint(archetypes.shape[0])]

# Yields a car EXACTLY every cars_per_sec, if possible
def regular(random):
  cars_per_tick = FLAGS.cars_per_sec * FLAGS.rate
  ticks_per_car = round(1 / cars_per_tick)
  cars_per_tick_int = math.ceil(cars_per_tick)
  for i in itertools.count(0):
    if ticks_per_car == 0 or i % ticks_per_car == 0:
      for _ in range(cars_per_tick_int):
        yield archetypes[0]
      yield None
    else: yield None

# Mindlessly copied and pasted from Hackers Delight.
# Only need the first 4 bits.
@jit(uint32(uint32),nopython=True,nogil=True,cache=True)
def inv_popcount(inv_i):
  i = (inv_i ^ 0xffffffff) & 0b1111
  i = i - ((i >> 1) & 0x55555555)
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
  return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24

@jit(void(int32[:],int32[:],float32,int32[:],float32[:,:,:],int32[:],int32[:],
  float32,int32[:],int32[:],float32[:],int32[:],int32[:]),
  nopython=True,nogil=True,cache=True)
def move_cars(dests,phases,length,nexts,state,leading,lastcar,rate,current_phase,elapsed,
    rewards,waiting,detected):
  update_lights(dests,phases,length,nexts,state,leading,lastcar,current_phase,elapsed)
  for e in range(leading.shape[0]):
    if leading[e] == lastcar[e]: continue
    if leading[e] < lastcar[e]:
      dv = sim(rate, state[e,:,leading[e]:lastcar[e]], state[e,:,leading[e]+1:lastcar[e]+1])
      if DECEL_PENALTY and dests[e] >= 0:
        rewards[dests[e]] += np.sum((dv < 0)) / 10
      if dests[e] >= 0:
        waiting[e] = np.sum((state[e,vi,leading[e]+1:lastcar[e]+1] < THRESH).astype(np.int32))
        detected[e] = np.sum((state[e,xi,leading[e]+1:lastcar[e]+1] > (length - 10)).astype(np.int32))
    else:
      state[e,:,0] = state[e,:,-1]
      dv = sim(rate, state[e,:,leading[e]:-1], state[e,:,leading[e]+1:])
      dv2 = sim(rate, state[e,:,:lastcar[e]], state[e,:,1:lastcar[e]+1])
      if DECEL_PENALTY and dests[e] >= 0:
        rewards[dests[e]] += (np.sum(dv < 0) + np.sum(dv2 < 0)) / 10
      if dests[e] >= 0:
        waiting[e] = np.sum((state[e,vi,leading[e]+1:] < THRESH).astype(np.int32))
        waiting[e] += np.sum((state[e,xi,1:lastcar[e]+1] > THRESH).astype(np.int32))
        detected[e] = np.sum((state[e,xi,leading[e]+1:] > (length - 10)).astype(np.int32))
        detected[e] += np.sum((state[e,xi,1:lastcar[e]+1:] > (length - 10)).astype(np.int32))

@jit(int32[:](int32[:],int32[:]),nopython=True,nogil=True,cache=True)
def cars_on_roads(leading, lastcar):
  inverted = (leading > lastcar).astype(np.int32)
  unwrapped_lastcar = inverted * np.int32(CAPACITY - 1) + lastcar
  return unwrapped_lastcar - leading

# Gym environment for the intelligent driver model
class TrafficEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def _step(self, action):
    if FLAGS.learn_switch:
      change = action
      self.current_phase[:] = np.logical_xor(self.current_phase, action) 
    else:
      change = np.logical_xor(self.current_phase, action).astype(np.int32) 
      self.current_phase[:] = action
    self.elapsed += 1 
    self.elapsed *= np.logical_not(change).astype(np.int32)
    self.rewards[:] = 0
    self.passed[:] = 0
    overflowed = self.add_new_cars(self.steps)
    move_cars(self.graph.dest, self.graph.phases, self.graph.len,
        self.graph.nexts, self.state, self.leading, self.lastcar, FLAGS.rate,
        self.current_phase, self.elapsed, self.rewards,
        self.waiting, self.detected)
    if FLAGS.mode == 'validate':
      adv = advance_hack(self.graph.dest, self.graph.len, self.graph.nexts, self.state,
        self.leading,self.lastcar,self.passed,self.rewards,self.passed_dst,self.trip_times, self.steps)
    else:
      adv = advance_finished_cars(self.graph.dest, self.graph.len, self.graph.nexts, self.state,
        self.leading,self.lastcar,self.passed,self.rewards,self.passed_dst)
    overflowed = adv or overflowed
    self.steps += 1
    return self.obs, self.rewards, overflowed, None

  def seed_generator(self, seed=None):
    self.rand = np.random.RandomState(seed)
    if FLAGS.poisson: self.rand_car = poisson(self.rand)
    else: self.rand_car = regular(self.rand)

  def cars_on_roads(self):
    return np.transpose(np.reshape(cars_on_roads(self.leading, self.lastcar)[:self.graph.train_roads],
      [4, self.graph.m, self.graph.n]), (1,2,0))

  def _reset(self):
    self.steps = np.float32(0)
    self.generated_cars = 0
    self.state[:,:,1] = 0 
    self.state[:,xi,1] = np.inf
    self.elapsed[:] = 0
    self.passed[:] = 0
    self.leading[:] = 1
    self.lastcar[:] = 1
    self.passed_dst[:] = False
    self.current_phase[:] = self.action_space.sample()
    self.passed[:] = 0
    self.waiting[:] = 0
    return self.obs

  def add_new_cars(self, tick):
    overflowed = False
    car = next(self.rand_car)
    while car is not None:
      self.generated_cars += 1
      car[wi] = tick
      overflowed = add_car(self.rand.choice(self.graph.entrypoints),car,self.state,
        self.leading,self.lastcar,self.rewards,self.graph.dest) or overflowed
      car = next(self.rand_car)
    return overflowed

  def init_viewer(self):
    from gym.envs.classic_control import rendering
    import pyglet.gl as gl

    # GL_LINES wrapper
    class Lines(rendering.Geom):
      def __init__(self, vs):
        rendering.Geom.__init__(self)
        self.vs = vs
        self.linewidth = rendering.LineWidth(1)
        self.add_attr(self.linewidth)

      def render1(self):
        gl.glBegin(gl.GL_LINES)
        for p in self.vs: gl.glVertex3f(p[0],p[1],0)
        gl.glEnd()

      def set_linewidth(self, x):
        self.linewidth.stroke = x

    max_x, max_y = np.max(self.graph.locs, axis=(0,1))
    min_x, min_y = np.min(self.graph.locs, axis=(0,1))
    self.viewer = rendering.Viewer(800, 800)
    self.viewer.set_bounds(min_x, max_x, min_y, max_y)
    self.roadlines = [rendering.Line(l[0],l[1]) for l in self.graph.locs]
    self.cars = [Lines([(0,0),(2,0)]) for _ in range(self.graph.roads)]
    self.roadrots = [rendering.Transform(translation=l[0], rotation=
      get_rot(l, self.graph.len)) for i, l in enumerate(self.graph.locs)]
    for r,c in zip(self.roadrots, self.cars): c.add_attr(r)
    for l in self.roadlines:
      l.set_color(0,1,0)
      self.viewer.add_geom(l)
    for c in self.cars:
      c.set_linewidth(5)
      c.set_color(0,0,1)
      self.viewer.add_geom(c)

  def _render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return
    if self.viewer is None:
      self.init_viewer()
    self.update_colors()
    self.update_locs()
    time.sleep(FLAGS.rate / 2)
    return self.viewer.render(return_rgb_array= mode=='rgb_array')

  def update_colors(self):
    for i in range(self.graph.train_roads):
      if self.graph.phases[i] == self.current_phase[self.graph.dest[i]]:
        if self.elapsed[self.graph.dest[i]] < YELLOW_TICKS:
          self.roadlines[i].set_color(1,1,0)
        else:
          self.roadlines[i].set_color(1,0,0)
      else:
        if self.elapsed[self.graph.dest[i]] < YELLOW_TICKS:
          self.roadlines[i].set_color(1,0,0)
        else:
          self.roadlines[i].set_color(0,1,0)

  def update_locs(self):
    for i in range(self.graph.roads):
      if self.leading[i] > self.lastcar[i]:
        xs,lens = np.hstack([
          self.state[i,[xi,li],self.leading[i]+1:],
          self.state[i,[xi,li],1:self.lastcar[i]+1]])
      else:
        xs,lens = self.state[i,[xi,li],self.leading[i]+1:self.lastcar[i]+1]
      if xs.shape[0] > 0:
        vals = np.concatenate(np.column_stack((xs, xs - lens)))
        self.cars[i].vs = np.column_stack((vals, np.zeros(vals.shape[0])))
      else: self.cars[i].vs = []
        
  def set_graph(self, graph):
    self.viewer = None
    self.graph = graph
    self.state = np.empty((graph.roads, params, CAPACITY), dtype=np.float32)
    self.leading = np.empty(graph.roads, dtype=np.int32)
    self.lastcar = np.empty(graph.roads, dtype=np.int32)
    self.action_space = GSpace([graph.intersections], np.int32(2))
    r = graph.train_roads
    i = graph.intersections
    obs_shape = [2*r+2*i]
    self.observation_space = GSpace(obs_shape, np.int32(1))
    self.obs = np.zeros(obs_shape, dtype=np.int32)
    self.passed = self.obs[:r]
    self.detected = self.obs[r:r+r]
    self.current_phase = self.obs[r+r:r+r+i]
    self.elapsed = self.obs[-i:]
    self.waiting = np.empty(r, dtype=np.int32)
    self.rewards = np.empty(i, dtype=np.float32)
    self.reward_size = self.rewards.size
    self.passed_dst = np.zeros(i,dtype=np.bool8)
    self.trip_times = []
    self.reset_entrypoints()

  def remi_reward(self):
    remi(self.graph.dest,self.graph.phases,self.current_phase,self.rewards,
      self.passed_dst,self.waiting)
    return self.rewards

  def reset_entrypoints(self):
    if FLAGS.entry == "random": spec = np.random.randint(0b1111, dtype='uint32')
    elif FLAGS.entry == "one": spec = 0b1110
    else: spec = 0
    self.graph.generate_entrypoints(spec)
    FLAGS.cars_per_sec = FLAGS.local_cars_per_sec * self.graph.m * inv_popcount(spec)
