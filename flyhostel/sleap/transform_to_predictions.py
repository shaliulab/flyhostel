import logging
import argparse
from sleap.io.dataset import Labels
from sleap.instance import Instance, Point
logger=logging.getLogger(__name__)

def prediction_to_gt(copy_instance):
    """
    From: sleap.gui.commands.make_instance_from_predicted_instance
    Docstring for prediction_to_gt
    
    :param copy_instance: Description
    :return: Description
    :rtype: Any
    """
    new_instance = Instance(
        skeleton=copy_instance.skeleton,
        from_predicted=copy_instance,
        frame=copy_instance.frame,
    )

    # go through each node in skeleton
    for node in new_instance.skeleton.node_names:
        # if we're copying from a skeleton that has this node
        if node in copy_instance and not copy_instance[node].isnan():
            # just copy x, y, and visible
            # we don't want to copy a PredictedPoint or score attribute
            new_instance[node] = Point(
                x=copy_instance[node].x,
                y=copy_instance[node].y,
                visible=copy_instance[node].visible,
                complete=False,
            )

    # copy the track
    new_instance.track = copy_instance.track

    return new_instance

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels")
    ap.add_argument("--output")
    args=ap.parse_args()

    labels=Labels.load_file(args.labels)

    for instance in labels.predicted_instances:
        user_instance=prediction_to_gt(instance)
        labels.add_instance(frame=user_instance.frame, instance=user_instance)
        labels.remove_instance(frame=instance.frame, instance=instance)
    labels.save_file(labels=labels, filename=args.output)

if __name__ == "__main__":
    main()
