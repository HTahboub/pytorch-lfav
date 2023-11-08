from glob import glob
from http.client import IncompleteRead
from time import sleep
from urllib.error import HTTPError

import youtube_dl
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
from youtube_dl.utils import DownloadError

DATA_PATH = "/work/vig/Datasets/LFAV/annotations/{split}/{split}_visual_weakly.csv"
SAVE_PATH = "/work/vig/Datasets/LFAV/videos/{split}"

# get video ids
ids = {"train": set(), "test": set(), "val": set()}
for split in ["train", "test", "val"]:
    with open(DATA_PATH.format(split=split), "r") as f:
        f.readline()
        for line in f:
            video_id = line.split("\t")[0]
            ids[split].add(video_id)

# print stats
for split in ["train", "test", "val"]:
    print(f"{split}: {len(ids[split])} videos")

# download videos
deleted = {"train": 0, "test": 0, "val": 0}
for split in ["train", "test", "val"]:
    for video_id in ids[split]:
        # in case we are running again
        if glob(SAVE_PATH.format(split=split) + f"/{video_id}.*"):
            continue
        while True:
            try:
                yt = YouTube(
                    f"https://www.youtube.com/watch?v={video_id}",
                    use_oauth=True,
                    allow_oauth_cache=True,
                )
                # download the video
                yt.streams.filter(progressive=True, file_extension="mp4").first().download(
                    SAVE_PATH.format(split=split),
                    filename=video_id + ".mp4"
                )
                break
            except VideoUnavailable:
                deleted[split] += 1
                print("Total unavailable:", deleted)
                break
            except (HTTPError, IncompleteRead):
                print("Connection error. Trying again...")
                sleep(10)
            except KeyError:
                print("Key error at video", video_id)
                try:
                    with youtube_dl.YoutubeDL(
                        {
                            "format": "bestvideo",
                            "outtmpl": "%(id)s.%(ext)s",
                        }
                    ) as ydl:
                        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                except DownloadError:
                    deleted[split] += 1
                    print("Total unavailable:", deleted)
                    break
            except Exception as e:
                print("Unexpected error at video", video_id)
                raise e
                # break  # we'll deal with these later

print("Deleted videos:")
for split in ["train", "test", "val"]:
    print(split + ":", deleted[split])
