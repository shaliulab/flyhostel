import os.path
from flyhostel.quantification.constants import FLYHOSTEL_ID
from flyhostel.utils import add_suffix

class DataView:

    def __init__(self, experiment, name, data, fig=None):
        self.experiment=experiment
        self.name = name
        self.data = data
        self.fig = fig

        self._csv_path = os.path.join(f"{experiment}_{name}.csv")
        self._fig_path = os.path.join(f"{experiment}_{name}.png")

    def save(self, output, suffix):
        columns = self.data.columns.tolist()
        columns.pop(columns.index(FLYHOSTEL_ID))
        columns.insert(0, FLYHOSTEL_ID)

        data = self.data[columns]

        data.to_csv(
            add_suffix(os.path.join(output, self._csv_path), suffix)
        )

        if self.fig is not None:
            self.fig.savefig(
                add_suffix(os.path.join(output, self._fig_path), suffix),
                transparent=False
            )
            self.fig.clear()
