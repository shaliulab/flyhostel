from collections import OrderedDict, defaultdict
from itertools import combinations, product
import logging
from typing import Iterable, Iterator, List, Tuple, Dict
logger=logging.getLogger(__name__)

def group_by_experiment(
    ids: Iterable[str], prefix_len: int = 26
) -> Dict[str, List[str]]:
    """
    Groups ids by their experiment prefix (first `prefix_len` chars).
    Preserves the first-seen order of experiments and of ids within each experiment.
    """
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for _id in ids:
        exp = _id[:prefix_len]
        groups.setdefault(exp, []).append(_id)
    return groups

def real_groups_iter(
    ids: Iterable[str], prefix_len: int = 26
) -> Iterator[Tuple[str, List[str]]]:
    """
    Yields (experiment_id, [ids in that experiment]) for each experiment.
    """
    for exp, lst in group_by_experiment(ids, prefix_len).items():
        yield exp, lst

def virtual_combinations_iter(
    ids: Iterable[str], prefix_len: int = 26
) -> Iterator[Tuple[str, List[str]]]:
    """
    Yields ("virtual", [combo_of_ids]) where each combo:
      - has the same length as the within-experiment group size (k)
      - contains ids drawn from k *different* experiments (no two from the same experiment)

    If experiments aren't all the same size, raises a ValueError.
    """
    groups = group_by_experiment(ids, prefix_len)
    if not groups:
        return
        yield  # keeps function a generator even if it returns immediately

    # verify equal sizes
    sizes = {len(v) for v in groups.values()}
    if len(sizes) != 1:
        raise ValueError(f"Experiments have different sizes: {sorted(sizes)}")
    k = sizes.pop()

    if len(groups) < k:
        logger.warning("Not enough experiments of group_size = %s", k)

    # choose k distinct experiments, then pick one id from each (cartesian product)
    exps = list(groups.keys())
    for exp_combo in combinations(exps, k):
        # product gives all ways to pick one id from each chosen experiment
        for choice in product(*(groups[e] for e in exp_combo)):
            yield "virtual", list(choice)
