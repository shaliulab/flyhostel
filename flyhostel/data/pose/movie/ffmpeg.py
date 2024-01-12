import subprocess
import shlex

def annotate_behavior_in_video_ffmpeg(input_video, frame_idx, behaviors, output_video, fps, gui_progress=False):
    filter_str=""
    fontfile="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_option=f"fontfile='{fontfile}':"
    # font_option=""
    
    fn0=frame_idx[0]
    for i, behavior in enumerate(behaviors):
        start_time=(frame_idx[i]-fn0)/fps
        end_time=(frame_idx[i]+skip-fn0)/fps
        filter_str+=f"drawtext={font_option}text='{behavior}':x=W-tw-10:y=10:enable='between(t,{start_time},{end_time})',"

    with open("filters.txt", "w") as handle:
        handle.write(filter_str[:-1])   

    ffmpeg_cmd=f"ffmpeg -hide_banner -loglevel info -y -i {input_video} -filter_complex_script filters.txt {output_video}"
    ffmpeg_cmd=shlex.split(ffmpeg_cmd)
    p=subprocess.Popen(ffmpeg_cmd)
    p.wait()
