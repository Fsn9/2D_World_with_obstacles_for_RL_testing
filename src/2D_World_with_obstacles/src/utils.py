from math import exp, log
import numpy as np

class Ramp:
    def __init__(self, y2, y1, x2, x1):
        self.slope = (y2-y1) / (x2 - x1)
        self.intersect = y2 - self.slope * x2

    def __call__(self, x):
        return self.slope * np.array(x) + self.intersect

class Exponential:
    def __init__(self, y2, y1, x2, x1):
        self.growth = 1.0 / (x1 - x2) * log(y1 / y2)
        self.gain = y2 / np.exp(self.growth * x2)

    def __repr__(self):
        return 'Exponential function | Growth: '+str(self.growth)+', | Gain:'+str(self.gain)+'\n'
    def __call__(self, x):
        return self.gain * np.exp(self.growth * x)

class CenterOfGravity:
    def __init__(self, weights, x):
        self.weights = weights
        self.x = x

    def __call__(self):
        try:
            cog = sum([w * x for w,x in zip(self.weights, self.x)]) / sum(self.weights)
        except: 
            raise Exception('den:',sum(self.weights))
            cog = 0.0
        return cog

class Integrator:
    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = [(0.0, 0.0) for _ in range(self.capacity)]

    def __len__(self):
        return self.capacity

    def __call__(self):
        distance = 0.0
        for i in range(self.capacity):
            if i == self.capacity - 1:
                break
            x1, y1 = self.samples[i][0], self.samples[i][1]
            x2, y2 = self.samples[i+1][0], self.samples[i+1][1]
            distance += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    def __repr__(self):
        return str(self.samples)
        
    def reset(self):
        self.samples = [(0.0, 0.0) for _ in range(self.capacity)]

    def update(self, x, y):
        self.samples.pop(0)
        self.samples.append((x, y))

class StateSpace:
    def __init__(self, start, end, resolution, name):
        self.start = start
        self.end = end
        self.resolution = resolution
        self.name = name
        self.__bins = np.linspace(start = start, stop = end, num = resolution, endpoint = True)

    @property
    def state_space(self):
        return self.__bins.tolist()
    
    def __call__(self, x):
        return self.__bins[np.abs(x - self.__bins).argmin()]

    def __repr__(self):
        return str(self.__bins)