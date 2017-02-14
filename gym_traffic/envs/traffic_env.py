import gym
from gym_traffic.spaces.gspace import GSpace
import numpy as np
import tensorflow as tf
from numba import jit, void, float64, float32, int64, int32, uint32, boolean
import numba
import itertools
import time
import math

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('local_cars_per_sec', 0.1, 'Cars entering the system per second')
flags.DEFINE_float('rate', 0.5, 'Number of seconds between simulator ticks')
flags.DEFINE_boolean('poisson', True, 'Should we use a Poisson distribution?')
flags.DEFINE_string('entry', 'random', 'Where should cars enter from?')
flags.DEFINE_boolean('learn_switch', False, "Learn switches, not phases")

# Python attribute access is expensive. We hardcode these params
YELLOW_TICKS = 5
DECEL_PENALTY = False
OVERFLOW_PENALTY = 10
WAITING_PENALTY = 0
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
archetypes[0,vi] = 0.2
archetypes[0,ai] = 0.02
archetypes[0,deltai] = 4
archetypes[0,v0i] = 0.8
archetypes[0,li] = 0.08
archetypes[0,bi] = 0.06
archetypes[0,ti] = 2
archetypes[0,s0i] = 0.03

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
@jit(boolean(int32,float32[:],float32[:,:,:],int32[:],int32[:],float32,int32[:],
  float32[:],int32[:]),nopython=True,nogil=True,cache=True)
def add_car(road,car,state,leading,lastcar,tick,newcars,rewards,dests):
  pos = wrap(lastcar[road] + 1)
  start_pos = np.inf
  if lastcar[road] != leading[road]:
    start_pos = state[road,xi,lastcar[road]] - state[road,li,lastcar[road]] \
        - state[road,s0i,lastcar[road]]
  if pos != leading[road]:
    state[road,:,pos] = car
    state[road,wi,pos] = tick
    state[road,xi,pos] = min(state[road,xi,pos], start_pos)
    lastcar[road] = pos
    if dests[road] >= 0: newcars[road] += 1
  elif dests[road] >= 0:
    rewards[dests[road]] -= OVERFLOW_PENALTY
    return True
  return False

# Remove cars with x coordinates beyond their roads' lengths
@jit(boolean(int32[:],float32,int32[:],float32[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:],
  float32),nopython=True,nogil=True,cache=True)
def advance_finished_cars(dests,length,nexts,state,leading,lastcar,queued,newcars,rewards,tick):
  overflowed = False
  for e in range(nexts.shape[0]):
    while leading[e] != lastcar[e] and state[e,xi,wrap(leading[e]+1)] > length:
      newlead = wrap(leading[e]+1)
      newrd = nexts[e]
      if newrd >= 0:
        queued[e] -= 1
        rewards[dests[e]] += 1
        state[e,xi,newlead] -= length
        overflowed = add_car(newrd,state[e,:,newlead],state,
            leading,lastcar,tick,newcars,rewards,dests) or overflowed
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

@jit(boolean(int32[:],int32[:],float32,int32[:],float32[:,:,:],int32[:],int32[:],
  float32,int32[:],int32[:],int32[:],int32[:],float32[:],float32),
  nopython=True,nogil=True,cache=True)
def move_cars(dests,phases,length,nexts,state,leading,lastcar,rate,current_phase,elapsed,
    queued,newcars,rewards,tick):
  update_lights(dests,phases,length,nexts,state,leading,lastcar,current_phase,elapsed)
  for e in range(leading.shape[0]):
    if leading[e] == lastcar[e]: continue
    if leading[e] < lastcar[e]:
      dv = sim(rate, state[e,:,leading[e]:lastcar[e]], state[e,:,leading[e]+1:lastcar[e]+1])
      if DECEL_PENALTY and dests[e] >= 0:
        rewards[dests[e]] += np.sum((dv < 0)) / 10
    else:
      state[e,:,0] = state[e,:,-1]
      dv = sim(rate, state[e,:,leading[e]:-1], state[e,:,leading[e]+1:])
      dv2 = sim(rate, state[e,:,:lastcar[e]], state[e,:,1:lastcar[e]+1])
      if DECEL_PENALTY and dests[e] >= 0:
        rewards[dests[e]] += (np.sum(dv < 0) + np.sum(dv2 < 0)) / 10
  return advance_finished_cars(dests,length,nexts,state,leading,lastcar,queued,newcars,rewards,tick)

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
    self.newcars[:] = 0
    overflowed = self.add_new_cars(self.steps)
    overflowed = move_cars(self.graph.dest, self.graph.phases, self.graph.len,
        self.graph.nexts, self.state, self.leading, self.lastcar, FLAGS.rate,
        self.current_phase, self.elapsed, self.queued, self.newcars, self.rewards,
        self.steps) or overflowed
    self.steps += 1
    self.queued += self.newcars
    if WAITING_PENALTY:
      self.rewards -= np.reshape(np.sum(
        np.square(self.obs[4:8]), axis=0) * WAITING_PENALTY, -1)
    return self.obs, self.rewards, overflowed, None

  def seed_generator(self, seed=None):
    self.rand = np.random.RandomState(seed)
    if FLAGS.poisson: self.rand_car = poisson(self.rand)
    else: self.rand_car = regular(self.rand)

  def _reset(self):
    self.steps = np.float32(0)
    self.generated_cars = 0
    self.state[:,:,1] = 0 
    self.state[:,xi,1] = np.inf
    self.elapsed[:] = 0
    self.queued[:] = 0
    self.leading[:] = 1
    self.lastcar[:] = 1
    self.current_phase[:] = np.round(
      np.random.randn(self.graph.intersections) + 0.5)
    self.queued[:] = 0
    return self.obs

  def add_new_cars(self, tick):
    overflowed = False
    car = next(self.rand_car)
    while car is not None:
      self.generated_cars += 1
      overflowed = add_car(self.rand.choice(self.graph.entrypoints),car,self.state,
        self.leading,self.lastcar,np.float32(tick),self.newcars,
        self.rewards,self.graph.dest) or overflowed
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
    time.sleep(FLAGS.rate)
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
    self.state = np.empty((self.graph.roads, params, CAPACITY), dtype=np.float32)
    self.leading = np.empty(self.graph.roads, dtype=np.int32)
    self.lastcar = np.empty(self.graph.roads, dtype=np.int32)
    self.action_space = GSpace(np.ones(graph.intersections, dtype=np.int32) + 1)
    obs_limit = np.full((10, graph.m, graph.n), CAPACITY-2, dtype=np.int32)
    obs_limit[4,:,:] = 2
    self.observation_space = GSpace(obs_limit)
    self.obs = np.empty_like(obs_limit)
    self.newcars = self.obs[:4].reshape(-1)
    self.queued = self.obs[4:8].reshape(-1)
    self.current_phase = self.obs[8].reshape(-1)
    self.elapsed = self.obs[9].reshape(-1)
    self.rewards = np.empty(graph.intersections, dtype=np.float32)
    self.reward_size = self.rewards.size
    self.reset_entrypoints()

  def reset_entrypoints(self):
    if FLAGS.entry == "random": spec = np.random.randint(0b1111, dtype='uint32')
    elif FLAGS.entry == "one": spec = 0b1110
    else: spec = 0
    self.graph.generate_entrypoints(spec)
    FLAGS.cars_per_sec = FLAGS.local_cars_per_sec * self.graph.m * inv_popcount(spec)
