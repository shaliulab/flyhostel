from abc import abstractmethod
from lib2to3.pytree import Base

from scipy.spatial import distance
import numpy as np

from .behaviors import BEHAVIORS
from .constants import QUIESCENT_STATE, MOVING_STATE


class BaseAnimal:

    neighbor_threshold=0.2 #cm

    def __init__(self, idx, x0, y0, movement_bout_length, neighbor_threshold, probability_interaction_movement, probability_spontaneous_movement, s0=MOVING_STATE):
        """
        Arguments:

            * idx (int): Animal id
            * x0 (int): First position along x dimension
            * y0 (int): First position along y dimension
            * neighbor_threshold (int): Maximum distance between two animals considered to be neighboring each other, in pxels
            * movement_bout_length (int): Expected length of quiescence bouts in step count
            * probability_interaction_movement (float): Probability that the animal will start moving,
                given an interaction with the animal
            * probability_spontaneous_movement (float): Intrinsic probability that the animal will start moving,
                independent of environment
            * s0 (int): First activity state 
        """

        self.idx=idx,
        self.x = x0
        self.y = y0
        self.state=s0
        self.steps = []
        self._time_moving = 0
        self._time_not_moving = 0
        self.neighbor_threshold = neighbor_threshold
        self.probability_interaction_movement = probability_interaction_movement
        self.probability_spontaneous_movement = probability_spontaneous_movement
        self._movement_bout_length = movement_bout_length


    @abstractmethod
    def move(self):
        raise NotImplementedError

    @property
    def centroid(self):
        return (self.x, self.y)

    @property
    def time_moving(self):
        return self._time_moving

    @property
    def time_not_moving(self):
        return self._time_moving

    @property
    def movement_bout_length(self):
        return self._movement_bout_length


    def save(self):
        self.steps.append(self.state)
        if self.state==QUIESCENT_STATE:
            self._time_moving = 0
            self._time_not_moving += 1
        else:
            self._time_moving+=1
            self._time_not_moving = 0


    def is_neighbor(self, other):
        return self.distance(other) < self.neighbor_threshold

    def neighbors(self, animals):
        neighbors = []

        for other in animals:
            if self.is_neighbor(other):
                neighbors.append(other)

        return neighbors

    def distance(self, other):

        return distance.euclidean(
            self.centroid, other.centroid
        )

    def spontaneously_moves(self):
        x=np.random.uniform(low=0, high=1, size=1)
        if x < self.probability_spontaneous_movement:
            self.state = MOVING_STATE
        else:
            self.state = QUIESCENT_STATE

    def conditionally_moves(self, p):
        """
        Arguments:
            * p (float): Probability the animal starts moving
        """
        
        x=np.random.uniform(low=0, high=1, size=1)
        if x < p:
            self.state = MOVING_STATE
        else:
            self.state = QUIESCENT_STATE

    def is_moving(self):
        return self.state==MOVING_STATE

    def stop_moving(self):
        self.state=QUIESCENT_STATE

    def interact(self, other):
        """
        Perform a behavior with another individual,
        which yields a probability of movement as a response
        """
        return self.probability_interaction_movement


class SimpleAnimal(BaseAnimal):

    def move(self):
        pass



class BehaviorSensitiveProbabilityOfMovementMixin:
    
    def interact(self, other):
        """
        Arguments:
            * other (Animal): Another Animal instance

        Returns
            * p (float): Probability that the animal starts moving as a result of the interaction
        """

        if self.interacts_with(other):
            p=self.sample_probability(other)

        else:
            p = 0.0
        
        return p


class RandomlyMovingAnimalMixin:
    def move(self):
        length = np.sqrt(np.random.uniform(0, 1))
        angle = np.pi * np.random.uniform(0, 2)
        self.x = np.sin(angle) * length
        self.y = np.cos(angle) * length


class RandomlyMovingAnimal(SimpleAnimal, RandomlyMovingAnimalMixin):
    pass


class AnimalWithArousalThreshold(SimpleAnimal, RandomlyMovingAnimalMixin):

    def arousal_threshold(self):
        scaling_factor = 1.0
        return -np.log(self._movement_bout_length * scaling_factor)

    def interacts_with(self, other):
        return self.is_neighbor(other)

    def sample_behavior(self):
        probabilities = [behavior.probability for behavior in BEHAVIORS]
        behavior = np.random.choice(BEHAVIORS, p=probabilities)
        return behavior

    def sample_probability(self):
        behavior = self.sample_behavior()
        return float(behavior.intensity > self.arousal_threshold())
    
class AnimalSensitiveToGroupSize(AnimalWithArousalThreshold):

    def arousal_threshold(self, animals):
        threshold = super(AnimalSensitiveToGroupSize, self).arousal_threshold()    
        scaling_factor=1.0
        threshold + np.log(animals * scaling_factor)