import numpy as np

class SocialSleepModel:

    def __init__(self, name, animals, repetitions, time_steps):
        self.name = name
        self.animals = animals
        self.time_steps = time_steps
        self.repetitions = repetitions

    @property
    def timeseries(self):
        return np.stack([
            animal.steps for animal in self.animals
        ], axis=1)


    def reset(self):
        for individual in self.animals:
            individual.steps.clear()
            individual._time_moving = 0
            individual._time_not_moving = 0


    def simulate(self):
        for step in range(self.time_steps):
            for individual in self.animals:
                if individual.is_moving():
                    for neighbor in individual.neighbors(self.animals):
                        p = individual.interact(neighbor)
                        neighbor.conditionally_moves(p)                        
                else:
                    individual.spontaneously_moves()
                    # TODO
                    # If individual did not move, make it more likely
                    # for other individuals to stop moving?
                if individual.time_moving > individual.movement_bout_length:
                    individual.stop_moving()

                individual.save()
            
            for individual in self.animals:
                individual.move()


    def __str__(self):
        return f"{self.name}_{str(len(self.animals)).zfill(2)}"
