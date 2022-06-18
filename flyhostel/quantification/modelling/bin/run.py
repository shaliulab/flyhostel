from flyhostel.quantification.modelling.model import SocialSleepModel
from flyhostel.quantification.modelling.animal import SimpleAnimal, RandomlyMovingAnimal
from flyhostel.quantification.modelling.manager import SimulationManager
from flyhostel.quantification.modelling.analysis import SleepAnalyser
from flyhostel.quantification.modelling.parameters import load_parameters
from .parser import get_parser


def main(ap=None, args=None):
    if args is None:
        ap = get_parser(ap)        
        args = ap.parse_args()
    
    run(args.number_of_animals, args.time_steps, output=args.output, n_jobs=args.n_jobs)


def run(number_of_animals, time_steps, output, n_jobs):

    params = load_parameters()
    simple_animals = [SimpleAnimal(idx=idx, x0=0, y0=0, **params) for idx in range(number_of_animals)]
    moving_animals = [RandomlyMovingAnimal(idx=idx, x0=0, y0=0, **params) for idx in range(number_of_animals)]
    model1 = SocialSleepModel(name="default_model", animals=simple_animals, time_steps=time_steps, repetitions=10)
    model2 = SocialSleepModel(name="moving_animals_model", animals=moving_animals, time_steps=time_steps, repetitions=10)
    analyser=SleepAnalyser(min_time_immobile=300, time_window_length=10)
    manager = SimulationManager(models=[model1, model2], analyser=analyser, output=output, n_jobs=n_jobs)
    manager.run()