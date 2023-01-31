import os
import sys
import shutil
import numpy as np
from collections import defaultdict
import sklearn
import sklearn.metrics
from moviepy.editor import *


query_index = 746
latents = np.load('latents.npy')



query_frame = latents[query_index][np.newaxis,:]
matrix = sklearn.metrics.pairwise.cosine_similarity(query_frame, latents)
indices =  np.argsort(matrix[0])[::-1][:10]

print(matrix[:,indices])

print (indices)
query_result_folder = f'{query_index}_clips'
videofolder = 'clips_4s'
if not os.path.exists(query_result_folder):
    os.mkdir(query_result_folder)

interval_length = 0
start2interval = defaultdict(int)
# corner case
start2interval[indices[0]] = 1
for i in range(1, len(indices)):
    interval_length +=1
    if indices[i] - indices[i-1] !=1:
        start = indices[i - interval_length]
        start2interval[start] = interval_length
        # corner case
        start2interval[indices[i]] = 1
        interval_length = 0

for start, interval_length in start2interval.items():
    videonames = [f'{videofolder}/maushaus_clip{index}.mp4' for index in range(start, start + interval_length)]
    clips = []
    for videoname in videonames:
        clip = VideoFileClip(videoname)
        clips.append(clip)
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(os.path.join(query_result_folder, f'video_{start}_{interval_length}.mp4')) 

    
