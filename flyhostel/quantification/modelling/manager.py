import os.path
import warnings

import numpy as np
import joblib
from .constants import OUTPUT_FOLDER

class SimulationManager:

    def __init__(self, models, output, analyser, n_jobs=1):

        if output is None:
            output = OUTPUT_FOLDER

        self.models = models
        self.output = output
        self.n_jobs = n_jobs
        self._analyser = analyser
        if os.path.exists(output):
            warnings.warn(f"{output} already exists")
        os.makedirs(output, exist_ok=True)

    def run_simulation(self, i, model, output):
        model.reset()
        model.simulate()
        np.savetxt(
            os.path.join(output, f"simulation_{model}_{str(i).zfill(3)}.txt"),
            model.timeseries
        )

        dt_total = self._analyser.summarise(model)
        dt_total.to_csv(
            os.path.join(output, f"simulation_{model}_{str(i).zfill(3)}.csv"),
        )

    def run_model(self, model, output):

        joblib.Parallel(
            n_jobs=self.n_jobs
        )(joblib.delayed(self.run_simulation)(
            i, model, output,        
        )
            for i in range(model.repetitions)
        )  
    
    def run(self):
        for model in self.models:
            self.run_model(model, self.output)

        