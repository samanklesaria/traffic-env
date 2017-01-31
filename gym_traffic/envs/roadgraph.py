from numba import jit, void, float64, float32, int64, int32, int8
import numpy as np

# Return an array of road locations for a grid road network
@jit(float32[:,:,:](float64,int64,int64,int64,int64), nopython=True,nogil=True,cache=True)
def get_locs_gridroad(eps,m,n,v,roads):
    locs = np.empty((roads,2,2), dtype=np.float32)
    for i in range(roads):
        d = i // v
        li = i % v
        col = li % n
        row = li // n
        r = i - 4*v
        if d == 0: locs[i] = np.array(((col-1,row-eps),(col,row-eps)))
        elif d == 1: locs[i] = np.array(((col+1,row+eps),(col,row+eps)))
        elif d == 2: locs[i] = np.array(((col+eps,row-1),(col+eps,row)))
        elif d == 3: locs[i] = np.array(((col-eps,row+1),(col-eps,row)))
        elif r < n: locs[i] = np.array(((r-eps,0),(r-eps,-1)))
        elif r < n+m: locs[i] = np.array(((n-1,r-n-eps),(n,r-n-eps)))
        elif r < 2*n+m: locs[i] = np.array(((r-n-m+eps,m-1),(r-n-m+eps,m)))
        else: locs[i] = np.array(((0,r-2*n-m+eps),(-1,r-2*n-m+eps)))
    return locs

# A graph representing a 2D grid with no turns
class GridRoad:
    def __init__(self, m, n, l):
        self.len = l
        self.n = n
        self.m = m
        v = m*n
        self.train_roads = 4*v
        self.roads = self.train_roads + 2*n + 2*m
        self.intersections = v
        self.locs = l * get_locs_gridroad(0.02,m,n,v,self.roads)
        self.phases = (np.arange(self.roads) // v < 2).astype(np.int8)
        self.dest = np.empty(self.roads, dtype=np.int32)
        self.nexts = np.array(list(map(self.get_next, range(self.roads))), dtype=np.int32)
        for i in range(self.roads):
            self.dest[i] = i%v if i<4*v else -1

    # Pick a probability distribution on entrypoints
    def generate_entrypoints(self, choices):
        n = self.n
        m = self.m
        v = m * n
        emp = np.empty(0,dtype=np.int32)
        self.entrypoints = np.concatenate((
            n*np.arange(m) if (choices & 1) == 0 else emp,
            v+n*np.arange(1,m+1)-1 if ((choices >> 1) & 1) == 0 else emp,
            2*v+np.arange(n) if ((choices >> 2) & 1) == 0 else emp,
            3*v+n*(m-1)+np.arange(n) if ((choices >> 3) & 1) == 0 else emp)).astype(np.int32)
                
    # Return the road a car should go to after road i, or -1
    def get_next(self, i):
        v = self.intersections
        n = self.n
        m = self.m
        if i >= 4*v: return -1
        col = i % n
        row = (i % v) // n
        if i < v: return i+1 if col < n-1 else 4*v+n+row
        if i < 2*v: return i-1 if col > 0 else 4*v+2*n+m+row
        if i < 3*v: return i+n if row < m-1 else 4*v+n+m+col
        return i-n if row > 0 else 4*v+col

# Convert a signal defined on roads to one defined on intersections
@jit(float32[:](int32,int32[:],float32[:]), nopython=True,nogil=True,cache=True)
def by_intersection(intersections,dests,road_cars):
    result = np.zeros(intersections, dtype=np.float32)
    for i in range(road_cars.shape[0]):
        result[dests[i]] += road_cars[i]
    return result

