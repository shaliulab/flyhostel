import logging
import argparse
import re
from sleap.io.dataset import Labels
logger=logging.getLogger(__name__)


def video_matches_regex(video, regex):
    """
    Docstring for video_matches_regex
    
    :param video: Path to .mp4
    :param regex: regular expression
    :return: True if the regex hits a match in the video_path
    """
    hits=re.search(regex, video.backend.filename)
    return hits is not None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels")
    ap.add_argument("--regex")
    ap.add_argument("--output")
    ap.add_argument("--force", action="store_true", default=False)
    args=ap.parse_args()

    labels=Labels.load_file(args.labels)

    videos_to_remove=[]
    for video in labels.videos:
        if not video_matches_regex(video, args.regex):
            videos_to_remove.append(video)

    if len(labels.videos)==len(videos_to_remove):
        raise Exception("Would remove all videos. Fix regex")

    videos_with_instances=[]
    for instance in labels.user_instances:
        if instance.video in videos_to_remove:
            videos_with_instances.append(instance.video)

    if not args.force:
        for video in videos_with_instances:
            logger.warning("Would remove user made instances in %s", video)

        if len(videos_with_instances)>0:
            raise Exception("Force mode disabled")

    for video in videos_to_remove:
        labels.remove_video(video)

    labels.save_file(labels=labels, filename=args.output)

if __name__ == "__main__":
    main()