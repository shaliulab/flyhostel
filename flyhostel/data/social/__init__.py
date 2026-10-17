"""
flyhostel.data.social — interactions between flies of a FlyHostel group.

                          FlyHostelLoaders (one experiment)
                                        |
                tracks.load_group_tracks: centroids on one grid, (T, N, 2) mm
                                        |
              movement.compute_movement: speed, immobility, sleep bouts
                                        |
   PIPELINE 1 (centroids)               |               PIPELINE 2 (pose)
   centroid.proximity_bouts(2.5 mm)     |     centroid.proximity_bouts(gate 5.5 mm)
                |                       |                      |
                |                       |     contact.detect_contacts (capsule skeleton,
                |                       |       pose read only inside gate windows)
                +-----------+-----------+----------------------+
                            |
          centroid.annotate_approach + orient: initiator / recipient
                            |
          classify.classify_bouts: static_contact | approach_to_immobile |
                                   mutual_encounter | approach_to_resting
                            |
          approach_response.approach_response: recipient response grade
                (none / limb / postural / locomotor) and initiator sleep latency

   passby.passby_events (centroids): an active fly enters and leaves a radius_mm circle
                around a focal fly (asleep / immobile / awake / any), with the sleep-wake
                state durations of both flies before and after

Entry points:
    from flyhostel.data.social import run_experiment
    tables = run_experiment(loaders, pose=True)

    python -m flyhostel.data.social --experiment FlyHostel1_6X_2025-10-02_16-00-00 --pose
"""
from .config import SocialConfig
from .pipeline import run, run_experiment
