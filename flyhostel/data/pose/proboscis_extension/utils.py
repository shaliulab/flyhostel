def read_experiment_list(path):
    """One experiment per line; blanks and #comments ignored."""
    with open(path) as handle:
        return [ln.strip() for ln in handle
                if ln.strip() and not ln.lstrip().startswith("#")]