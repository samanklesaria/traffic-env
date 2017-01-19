import gym
from gym import error, spaces, utils
import numpy as np
import tensorflow as tf
from numba import jit, jitclass, deferred_type, void, float64, float32, int64, int32, int8
import numba
from gym.envs.classic_control import rendering
from gym_traffic.envs.roadgraph import grid, GridRoad
from pyglet.gl import *
import time

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('local_cars_per_sec', 0.3, 'Cars entering the system per second')
flags.DEFINE_float('rate', 0.1, 'Number of seconds between simulator ticks')

# GL_LINES wrapper
class Lines(rendering.Geom):
  def __init__(self, vs):
    rendering.Geom.__init__(self)
    self.vs = vs
    self.linewidth = rendering.LineWidth(1)
    self.add_attr(self.linewidth)

  def render1(self):
    glBegin(GL_LINES)
    for p in self.vs: glVertex3f(p[0],p[1],0)
    glEnd()

  def set_linewidth(self, x):
    self.linewidth.stroke = x

# Get the rotation of a line segment
def get_rot(line, length):
  l = line / length
  return np.arctan2(l[1,1] - l[0,1], l[1,0] - l[0,0])

# New cars have parameters sampled uniformly from archetypes
params = 9
xi, vi, li, ai, deltai, v0i, bi, ti, s0i = range(params)
archetypes = np.zeros((1, params))
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
@jit(int32(int32), nopython=True, nogil=True)
def wrap(a): return 1 if a >= CAPACITY else a

# The Intelligent Driver Model of lateral acceleration
@jit(void(float32,float32[:,:],float32[:,:]), nopython=True, nogil=True)
def sim(r, ld, me):
  v = me[vi]
  s_star = me[s0i] + np.maximum(0, v*me[ti] + v *
          (v - ld[vi]) / (2 * np.sqrt(me[ai]*me[bi])))
  s = ld[xi] - me[xi] - ld[li]
  dv = me[ai] * (1 - (v / me[v0i])**me[deltai] - np.square(s_star / (s + EPS)))
  dx = r*v + 0.5*dv*np.square(r)
  me[xi] += (dx > 0)*dx
  me[vi] = np.maximum(0, v + dv*r)

# Update the leading car at the end of each road depending on light phases
@jit(nopython=True, nogil=True)
def update_lights(graph, state, leading, lastcar, current_phase):
  for e in range(graph.train_roads):
    if graph.phases[e] == current_phase[graph.dest[e]]:
      state[e, xi, leading[e]] = graph.length(e)
    else:
      newrd = graph.next(e)
      if newrd >= 0 and lastcar[newrd] != leading[newrd]:
        state[e, xi, leading[e]] = state[newrd, xi, lastcar[newrd]]
        state[e, xi, leading[e]] += graph.length(e)
      else:
        state[e, xi, leading[e]] = np.inf

# Add a new car to a road
@jit(nopython=True, nogil=True)
def add_car(road, car, state, leading, lastcar):
  pos = wrap(lastcar[road] + 1)
  start_pos = np.inf
  if lastcar[road] != leading[road]:
    start_pos = state[road,xi,lastcar[road]] - state[road,li,lastcar[road]] \
        - state[road,s0i,lastcar[road]]
  if pos != leading[road]:
    state[road,:,pos] = car
    state[road,xi,pos] = min(state[road,xi,pos], start_pos)
    lastcar[road] = pos

# Remove cars with x coordinates beyond their roads' lengths
@jit(nopython=True, nogil=True)
def advance_finished_cars(graph, state, leading, lastcar, counts):
  counts[:] = 0
  for e in range(graph.roads):
    while leading[e] != lastcar[e] and state[e,xi,wrap(leading[e]+1)] > graph.length(e):
      newlead = wrap(leading[e]+1)
      newrd = graph.next(e)
      if newrd >= 0:
        state[e,xi,newlead] -= graph.length(e)
        add_car(newrd, state[e,:,newlead], state, leading, lastcar)
        counts[graph.dest[e]] += 1
      state[e,:,newlead] = state[e,:,leading[e]]
      leading[e] = newlead

# Get the number of cars on each road
@jit(nopython=True, nogil=True)
def cars_on_roads(leading, lastcar):
  inverted = (leading > lastcar).astype(np.int32)
  unwrapped_lastcar = (inverted * (CAPACITY - 1)).astype(np.int32) + lastcar
  return unwrapped_lastcar - leading

# Yields None separated groups of incoming cars for each tick
def poisson(random):
  cars_per_tick = FLAGS.cars_per_sec * FLAGS.rate
  while True:
    for _ in range(int(random.exponential(1/cars_per_tick))): yield None
    yield archetypes[random.randint(archetypes.shape[0])]

@jit(nopython=True, nogil=True)
def move_cars(graph, state, leading, lastcar, rate, current_phase, counts):
  update_lights(graph, state, leading, lastcar, current_phase)
  for e in range(graph.roads):
    if leading[e] == lastcar[e]: continue
    if leading[e] < lastcar[e]:
      sim(rate, state[e,:,leading[e]:lastcar[e]],
            state[e,:,leading[e]+1:lastcar[e]+1])
    else:
        state[e,:,0] = state[e,:,-1]
        sim(rate, state[e,:,leading[e]:-1],
            state[e,:,leading[e]+1:])
        sim(rate, state[e,:,:lastcar[e]],
            state[e,:,1:lastcar[e]+1])
  advance_finished_cars(graph, state, leading, lastcar, counts)
  return cars_on_roads(leading, lastcar)[:graph.train_roads]


# Gym environment for the intelligent driver model
class TrafficEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def _step(self, action):
    self.current_phase = np.array(action).astype(np.int8)
    if self.adding_steps is None or self.steps < self.adding_steps: self.add_new_cars()
    self.steps += 1
    current_cars = move_cars(self.graph, self.state, self.leading,
        self.lastcar, FLAGS.rate, self.current_phase, self.counts)
    return current_cars, self.counts, self.steps >= self.total_steps, None

  def _reset(self):
    self.steps = 0
    self.state[:,:,1] = 0 
    self.state[:,xi,1] = np.inf
    self.current_phase = np.round(
      np.random.randn(self.graph.intersections) + 0.5).astype(np.int8)
    self.rand_car = poisson(np.random.RandomState())
    self.leading = np.ones(self.graph.roads, dtype=np.int32)
    self.lastcar = np.ones(self.graph.roads, dtype=np.int32)
    if self.randomized: self.graph.generate_entrypoints(np.random.randint(0b1111))
    return cars_on_roads(self.leading, self.lastcar)[:self.graph.train_roads]

  def add_new_cars(self):
    car = next(self.rand_car)
    while car is not None:
      add_car(np.random.choice(self.graph.entrypoints), car, self.state, self.leading, self.lastcar)
      car = next(self.rand_car)

  def init_viewer(self):
    max_x, max_y = np.max(self.graph.locs, axis=(0,1))
    min_x, min_y = np.min(self.graph.locs, axis=(0,1))
    self.viewer = rendering.Viewer(600, 600)
    self.viewer.set_bounds(min_x, max_x, min_y, max_y)
    self.roadlines = [rendering.Line(l[0],l[1]) for l in self.graph.locs]
    self.cars = [Lines([(0,0),(2,0)]) for _ in range(self.graph.roads)]
    self.roadrots = [rendering.Transform(translation=l[0], rotation=
      get_rot(l, self.graph.length(i))) for i, l in enumerate(self.graph.locs)]
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
        
  def set_graph(self, graph, total_steps, cooldown=0, randomized=False):
    self.viewer = None
    self.randomized = randomized
    if not randomized: graph.generate_entrypoints(0)
    self.graph = graph
    self.state = np.empty((self.graph.roads, params, CAPACITY), dtype=np.float32)
    self.action_space = spaces.MultiDiscrete([[0,1]] * graph.intersections)
    self.observation_space = spaces.Box(low=0, high=CAPACITY-2, shape=graph.train_roads)
    self.counts = np.empty(graph.intersections, dtype=np.float32)
    self.total_steps = total_steps
    self.adding_steps = total_steps - cooldown
    self._reset()
