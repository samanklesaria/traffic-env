import argparse
import importlib

PARSER = argparse.ArgumentParser()
PARSER.defaults = {}
PARSER.derivations = []

class Flags(argparse.Namespace):
  def __getattr__(self, name):
    if name in self.__dict__: return self.__dict__[name]
    elif name in PARSER.defaults: return PARSER.defaults[name]
    else: raise AttributeError(name)

FLAGS = Flags()

def parse_flags():
  PARSER.parse_args(namespace=FLAGS)
  apply_derivations(PARSER)

def add_argument(name, default, **kwargs):
  PARSER.add_argument(name, **kwargs)
  PARSER.defaults[name.replace('-','')] = default

def update_flags(**kwargs):
  FLAGS.__dict__.update(**kwargs)
  apply_derivations(PARSER)

def apply_derivations(PARSER):
  for _ in range(10):
    old_dict = FLAGS.__dict__.copy()
    for f in PARSER.derivations: f()
    if FLAGS.__dict__ == old_dict: return
  raise Exception("Could not find settings fixed point")

def add_derivation(f):
  PARSER.derivations.append(f)
