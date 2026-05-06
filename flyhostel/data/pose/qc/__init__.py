"""
A module to generate a quality control report of the experiment

Report technical aspects of the experiment that could compromise the validity of the results

1) time passed between chunks of the experiment:
  if a lot of time passes between the GPU finishing the encoding of a chunk
  and the starting of the next one, the flies may move and if there is no marker
  or > 1 fly, their identity may be untraceable

2) time passed between frames of the same chunk:
  similar to 1) but between any two frames in the same chunk
"""