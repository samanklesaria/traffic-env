import gym
from gym import error, spaces, utils
import numpy as np
import tensorflow as tf
from numba import jit, jitclass, deferred_type, void, float64, float32, int64, int32, int8
import numba
import itertools
import time
import math

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('local_cars_per_sec', 0.2, 'Cars entering the system per second')
flags.DEFINE_float('rate', 0.5, 'Number of seconds between simulator ticks')
flags.DEFINE_boolean('poisson', True, 'Should we use a Poisson distribution?')
flags.DEFINE_boolean('obs_deltas', True, 'Should we observe car entries, not cars?')
flags.DEFINE_boolean('reward_time', False, 'Should we reward negative trip times')
flags.DEFINE_boolean('decel_penalty', False, 'Should we penalize decelleration')
flags.DEFINE_string('entry', 'all', 'Where should cars enter from?')
flags.DEFINE_float('overflow_penalty', 0, 'Overflow penalty')

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
archetypes[0,s0i] = 0.01

CAPACITY = 20
EPS = 1e-8

# Like mod, but preserves index 0
@jit(int32(int32), nopython=True,nogil=True,cache=True)
def wrap(a): return 1 if a >= CAPACITY else a

# The Intelligent Driver Model of lateral acceleration
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
@jit(void(int32[:],int8[:],float32,int32[:],float32[:,:,:],int32[:],int32[:],int8[:]),
    nopython=True,nogil=True,cache=True)
def update_lights(dests, phases, length, nexts, state, leading, lastcar, current_phase):
  for (e,dst) in enumerate(dests):
    if dst == -1: return
    if phases[e] == current_phase[dst]:
      state[e, xi, leading[e]] = length
    else:
      newrd = nexts[e]
      if newrd >= 0 and lastcar[newrd] != leading[newrd]:
        state[e, xi, leading[e]] = state[newrd, xi, lastcar[newrd]]
        state[e, xi, leading[e]] += length
      else:
        state[e, xi, leading[e]] = np.inf

# Add a new car to a road
@jit(void(int32,float32[:],float32[:,:,:],int32[:],int32[:],float32,float32[:],
  float32[:],float32,int32[:]),nopython=True,nogil=True,cache=True)
def add_car(road, car, state, leading, lastcar, tick, counts, rewards, penalty, dests):
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
    if dests[road] >= 0: counts[road] += 1
  elif dests[road] >= 0:
    rewards[dests[road]] -= penalty

# Remove cars with x coordinates beyond their roads' lengths
@jit(void(int32[:],float32,int32[:],float32[:,:,:],int32[:],int32[:],float32[:],float32[:],
  float32,float32,int8),nopython=True,nogil=True,cache=True)
def advance_finished_cars(dests,length,nexts,state,leading,lastcar,counts,rewards,penalty,tick,tb):
  for e in range(nexts.shape[0]):
    while leading[e] != lastcar[e] and state[e,xi,wrap(leading[e]+1)] > length:
      newlead = wrap(leading[e]+1)
      newrd = nexts[e]
      if newrd >= 0:
        if tb: rewards[dests[e]] -= (tick - state[e,wi,newlead]) / 100
        else: rewards[dests[e]] += 1
        state[e,xi,newlead] -= length
        add_car(newrd,state[e,:,newlead],state,leading,lastcar,tick,counts,rewards,penalty,dests)
      state[e,:,newlead] = state[e,:,leading[e]]
      leading[e] = newlead

# Get the number of cars on each road
@jit(int32[:](int32[:],int32[:]),nopython=True,nogil=True,cache=True)
def cars_on_roads(leading, lastcar):
  inverted = (leading > lastcar).astype(np.int32)
  unwrapped_lastcar = inverted * np.int32(CAPACITY - 1) + lastcar
  return unwrapped_lastcar - leading

# Yields None separated groups of incoming cars for each tick according to Poisson
def poisson(random):
  cars_per_tick = FLAGS.cars_per_sec * FLAGS.rate
  while True:
    for _ in range(int(random.exponential(1/cars_per_tick))): yield None
    yield archetypes[random.randint(archetypes.shape[0])]

# Yields a car EXACTLY every cars_per_sec, if possible
def regular(random):
  cars_per_tick = FLAGS.cars_per_sec * FLAGS.rate
  ticks_per_car = int(1 / cars_per_tick)
  for i in itertools.count(0):
    if ticks_per_car == 0 or i % ticks_per_car == 0:
      for _ in range(math.ceil(cars_per_tick)):
        yield archetypes[0]
      yield None
    else: yield None

@jit(void(int32[:],int8[:],float32,int32[:],float32[:,:,:],int32[:],int32[:],
  float32,int8[:],float32[:],float32[:],float32,float32,int8,int8),nopython=True,nogil=True,cache=True)
def move_cars(dests,phases,length,nexts,state,leading,lastcar,rate,current_phase,counts,
    rewards,penalty,tick,dec_pen,time_rew):
  update_lights(dests,phases,length,nexts,state,leading,lastcar,current_phase)
  for e in range(leading.shape[0]):
    if leading[e] == lastcar[e]: continue
    if leading[e] < lastcar[e]:
      dv = sim(rate, state[e,:,leading[e]:lastcar[e]], state[e,:,leading[e]+1:lastcar[e]+1])
      if dec_pen and dests[e] >= 0: rewards[e] += np.sum(np.minimum(dv,0))
    else:
      state[e,:,0] = state[e,:,-1]
      dv = sim(rate, state[e,:,leading[e]:-1], state[e,:,leading[e]+1:])
      dv2 = sim(rate, state[e,:,:lastcar[e]], state[e,:,1:lastcar[e]+1])
      if dec_pen and dests[e] >= 0: rewards[e] += np.sum(np.minimum(dv,0)) + np.sum(np.minimum(dv2,0))
  advance_finished_cars(dests,length,nexts,state,leading,lastcar,counts,rewards,penalty,tick,time_rew)

# Gym environment for the intelligent driver model
class TrafficEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def _step(self, action):
    self.current_phase = np.array(action).astype(np.int8)
    self.rewards[:] = 0
    self.counts[:] = 0
    if self.adding_steps is None or self.steps < self.adding_steps:
      self.add_new_cars(self.steps)
    move_cars(self.graph.dest, self.graph.phases, self.graph.len,
        self.graph.nexts, self.state, self.leading,
        self.lastcar, FLAGS.rate, self.current_phase, self.counts, self.rewards,
        FLAGS.overflow_penalty, self.steps, FLAGS.decel_penalty, FLAGS.reward_time)
    self.steps += 1
    if FLAGS.obs_deltas:
      obs = self.counts
    else:
      obs = cars_on_roads(self.leading, self.lastcar)[:self.graph.train_roads]
    obs = np.reshape(obs,[4,self.graph.m, self.graph.n])
    if self.steps == self.total_steps and FLAGS.reward_time:
      remaining = np.reshape(cars_on_roads(self.leading, self.lastcar)[:self.graph.train_roads],
        [4,self.graph.m,self.graph.n])
      self.rewards -= np.reshape(np.sum(remaining, axis=0) * 30, -1)
    return obs, self.rewards, self.steps >= self.total_steps, None

  def _reset(self):
    self.steps = 0
    self.generated_cars = 0
    self.state[:,:,1] = 0 
    self.state[:,xi,1] = np.inf
    self.current_phase = np.round(
      np.random.randn(self.graph.intersections) + 0.5).astype(np.int8)
    if FLAGS.poisson: self.rand_car = poisson(np.random.RandomState())
    else: self.rand_car = regular(np.random.RandomState())
    self.leading = np.ones(self.graph.roads, dtype=np.int32)
    self.lastcar = np.ones(self.graph.roads, dtype=np.int32)
    if self.randomized: self.graph.generate_entrypoints(np.random.randint(0b1111))
    return np.reshape(cars_on_roads(self.leading, self.lastcar)[:self.graph.train_roads],
        [4, self.graph.m, self.graph.n])

  def add_new_cars(self, tick):
    car = next(self.rand_car)
    while car is not None:
      self.generated_cars += 1
      add_car(np.random.choice(self.graph.entrypoints),car,self.state,
        self.leading,self.lastcar,np.float32(tick),self.counts,self.rewards,
        FLAGS.overflow_penalty,self.graph.dest)
      car = next(self.rand_car)

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
    time.sleep(FLAGS.rate / 1.5)
    return self.viewer.render(return_rgb_array= mode=='rgb_array')

  def update_colors(self):
    for i in range(self.graph.train_roads):
      if self.graph.phases[i] == self.current_phase[self.graph.dest[i]]:
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
    self.randomized = FLAGS.entry == "random"
    if FLAGS.entry == "one": graph.generate_entrypoints(0b1110)
    elif FLAGS.entry == "all": graph.generate_entrypoints(0)
    self.graph = graph
    self.state = np.empty((self.graph.roads, params, CAPACITY), dtype=np.float32)
    self.action_space = spaces.MultiDiscrete([[0,1]] * graph.intersections)
    self.observation_space = spaces.Box(low=0, high=CAPACITY-2, shape=(4, graph.m, graph.n))
    self.counts = np.empty(graph.train_roads, dtype=np.float32)
    self.rewards = np.empty(graph.intersections, dtype=np.float32)
    self.total_steps = FLAGS.episode_ticks
    self.adding_steps = self.total_steps - int(FLAGS.cooldown * FLAGS.episode_ticks)
    self._reset()
