import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from decord import VideoReader
from decord import cpu

class VideoDataset(Dataset):
    def __init__(self, folder_path, img_size=256, seq_len=16, stride=1):
        self.folder_path = folder_path
        self.img_size = img_size
        self.seq_len = seq_len
        self.stride = stride
        self.data = []

        self.folder_list = os.listdir(folder_path)
        for item in self.folder_list:
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                self._process_image_folder(item_path)
            elif item.lower().endswith('.gif'):
                self._process_gif(item_path)
            elif item.lower().endswith('.mp4'):
                self._process_mp4(item_path)

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        ])

    def _process_image_folder(self, folder_path):
        files = os.listdir(folder_path)
        files.sort()
        files = [os.path.join(folder_path, f) for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        for i in range(0, len(files) - self.seq_len + 1, self.stride):
            self.data.append(('image', files[i:i+self.seq_len]))

    def _process_gif(self, gif_path):
        gif = Image.open(gif_path)
        frame_count = gif.n_frames
        for i in range(0, frame_count - self.seq_len + 1, self.stride):
            self.data.append(('gif', (gif_path, i, i+self.seq_len)))

    def _process_mp4(self, mp4_path):
        vr = VideoReader(mp4_path, ctx=cpu(0))
        frame_count = len(vr)
        for i in range(0, frame_count - self.seq_len + 1, self.stride):
            self.data.append(('mp4', (mp4_path, i, i+self.seq_len)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_type, item = self.data[idx]
        frames = []

        if data_type == 'image':
            for f in item:
                image = Image.open(f)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = self.transform(image)
                frames.append(image)
        elif data_type == 'gif':
            gif_path, start, end = item
            gif = Image.open(gif_path)
            for frame in range(start, end):
                gif.seek(frame)
                image = gif.copy()
                image = self.transform(image)
                frames.append(image)
        elif data_type == 'mp4':
            mp4_path, start, end = item
            vr = VideoReader(mp4_path, ctx=cpu(0))
            for frame_idx in range(start, end):
                image = vr[frame_idx].asnumpy()
                image = Image.fromarray(image)
                image = self.transform(image)
                frames.append(image)

        frames = torch.stack(frames, 0).permute(0, 3, 1, 2)  # S,C,H,W
        return frames

class VideoDatasetNP(Dataset):
    def __init__(self, video_list, img_size=256, seq_len=16, stride=1):
        self.video_list = video_list
        self.img_size = img_size
        self.data = []
        for video in self.video_list:
            for i in range(0, len(video)-seq_len + 1, stride):
                self.data.append(video[i:i+seq_len])

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        ])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        frames= self.data[idx]
        frames = torch.from_numpy(frames).permute(0,3,1,2) # S,C,H,W
        frames = self.transform(frames)
        return frames
