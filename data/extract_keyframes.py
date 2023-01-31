import os
import sys
import shutil
from moviepy.editor import *
import numpy as np
from collections import defaultdict



def calc_interval(cluster_labels):
    interval2start = defaultdict(list)
    interval_list = []
    interval_length = 0
    prev_label = cluster_labels[0]
    for i in range(1, len(cluster_labels)):
        interval_length +=1
        if cluster_labels[i] != cluster_labels[i-1]: 
            start = i - interval_length             
            prev_label = cluster_labels[i]
            interval2start[interval_length].append(start)
            interval_list.append(interval_length)
            interval_length = 0
            
    return interval2start

from sklearn.cluster import KMeans
latents = np.load('latents.npy')
kmeans = KMeans(n_clusters=10, random_state=0).fit(latents)


calc_interval(kmeans.labels_)

interval2start = calc_interval(kmeans.labels_)
sorted_intervals = sorted(list(interval2start.keys()))[::-1]
cluster2interval2start = defaultdict(list)


for interval_length in sorted_intervals:
    starts = interval2start[interval_length]
    for start in starts:
        cluster_id = kmeans.labels_[start]
        cluster2interval2start[cluster_id].append((start, interval_length))


videofolder = 'clips_4s'

for num, (cluster_id, lst) in enumerate(cluster2interval2start.items()):
    groupname = f'{videofolder}/group_{cluster_id}'
    for tu in lst:

        start, interval_length = tu        
        videonames = [f'{videofolder}/maushaus_clip{index}.mp4' for index in range(start, start + interval_length)]

        if not os.path.exists(groupname):
            os.mkdir(groupname)
        clips = []
        for videoname in videonames:
            clip = VideoFileClip(videoname)
            clips.append(clip)
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(os.path.join(groupname, f'video_{start}_{interval_length}.mp4'))
    if num > 10:
        break
