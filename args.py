import argparse

PARSER = argparse.ArgumentParser()
PARSER.defaults = {}

class Flags(argparse.Namespace):
  def __getattr__(self, name):
    global PARSER
    if name in self.__dict__: return self.__dict__[name]
    elif name in PARSER.defaults: return PARSER.defaults[name]
    else: raise AttributeError(name)

FLAGS = Flags()

def parse_flags():
  global FLAGS
  global PARSER
  PARSER.parse_args(namespace=FLAGS)

def add_argument(name, default, **kwargs):
  global PARSER
  global FLAGS
  PARSER.add_argument(name, **kwargs)
  PARSER.defaults[name.replace('-','')] = default

