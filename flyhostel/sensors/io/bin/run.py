import os.path
from flyhostel.sensors.io.io import load_data, save_data, plot_data
from .parser import get_parser

def main(args=None, ap=None):

    if args is None:
        if ap is None:
            ap = get_parser()

        args = ap.parse_args()

    data = load_data(
        store_path=args.input,
        reference_hour=args.reference_hour,
        threshold=args.light_threshold,
    )

    os.makedirs(args.output, exist_ok=True)

    experiment_date = os.path.basename(args.input.rstrip("/"))
    dest=os.path.join(
        args.output,
        f"{experiment_date}_environment-log.csv"
    )

    save_data(dest, data)
    root=os.path.join(
        args.output,
        f"{experiment_date}"
    )
    plot_data(root, data, title=experiment_date)

if __name__ == "__main__":
    main()