class Behavior:

    def __init__(self, name, intensity, probability):
        self.name = name
        self.intensity = intensity
        self.probability = probability


p_feeding=0.1
p_rejection=0.2
p_walking=0.4
p_flyover=0.01
p_none=1-p_feeding-p_rejection-p_walking-p_flyover

BEHAVIORS = [
    Behavior("none", 0, p_none),
    Behavior("feeding", 1, p_feeding),
    Behavior("rejection", 3, p_rejection),
    Behavior("walking", 1, p_walking),
    Behavior("flyover", 3, p_flyover),
]