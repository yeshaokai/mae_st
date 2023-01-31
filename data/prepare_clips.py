from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os


duration = 1
total_length = 59 * 60
num_clips = total_length // duration
clip_folder_name = f'clips_{duration}s'

if not os.path.exists(clip_folder_name):
    os.mkdir(clip_folder_name)
    

def cut_videos():
    long_video = 'maushaus.mp4'
    start_time = 0
    
    for i in range(num_clips):        
        end_time = start_time + duration
        ffmpeg_extract_subclip(long_video, start_time, end_time, targetname=f"{clip_folder_name}/maushaus_clip{i}.mp4")        
        start_time = end_time
def create_train_csv():
    with open(f'./{clip_folder_name}/trains.csv', 'w') as f:
        cwd = os.getcwd()
        for i in range(num_clips):
            f.write(os.path.join(cwd, f'{clip_folder_name}/maushaus_clip{i}.mp4') + ' ' + str(i) + '\n')

    with open(f'./{clip_folder_name}/tests.csv', 'w') as f:
        cwd = os.getcwd()
        for i in range(num_clips):
            f.write(os.path.join(cwd, f'{clip_folder_name}/maushaus_clip{i}.mp4') + ' ' + str(i) + '\n')            


create_train_csv()
cut_videos()
