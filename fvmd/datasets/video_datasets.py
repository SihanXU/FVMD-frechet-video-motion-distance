import torch
import os
import numpy as np
from PIL import Image
import cv2 
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VideoDataset(Dataset):
    def __init__(self, imagefolder_path, img_size=256, seq_len=16, stride=1):
        self.imagefolder_path = imagefolder_path
        self.folder_image_list = os.listdir(imagefolder_path)
        self.img_size = img_size
        self.seq_len = seq_len
        self.stride = stride
        self.data = []

        for item in self.folder_image_list:
            item_path = os.path.join(self.imagefolder_path, item)
            if os.path.isdir(item_path):
                files = os.listdir(item_path)
                files.sort()
                image_files = [os.path.join(item_path, f) for f in files if f.endswith((".jpg", ".png", ".jpeg"))]
                for i in range(0, len(image_files) - seq_len + 1, stride):
                    self.data.append(image_files[i:i+seq_len])
            elif item.endswith(".gif") or item.endswith(".mp4"):
                self.data.append(item_path)

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if isinstance(item, list):
            return self._load_image_sequence(item)
        elif item.endswith(".gif"):
            return self._load_gif(item)
        elif item.endswith(".mp4"):
            return self._load_mp4(item)

    def _load_image_sequence(self, image_list):
        frames = []
        for f in image_list:
            image = Image.open(f)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
            frames.append(np.array(image))
        frames = torch.from_numpy(np.stack(frames, 0)).permute(0, 3, 1, 2)  # S,C,H,W
        return frames

    def _load_gif(self, gif_path):
        gif = Image.open(gif_path)
        frames = []
        try:
            while True:
                frame = gif.copy().convert("RGB")
                frame = self.transform(frame)
                frames.append(np.array(frame))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        frames = torch.from_numpy(np.stack(frames, 0)).permute(0, 3, 1, 2)  # S,C,H,W
        if frames.size(0) < self.seq_len:
            padding = torch.zeros((self.seq_len - frames.size(0), *frames.shape[1:]))
            frames = torch.cat([frames, padding], dim=0)
        return frames[:self.seq_len]

    def _load_mp4(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.seq_len and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(np.array(frame))

        cap.release()
        frames = torch.from_numpy(np.stack(frames, 0)).permute(0, 3, 1, 2)  # S,C,H,W
        if frames.size(0) < self.seq_len:
            padding = torch.zeros((self.seq_len - frames.size(0), *frames.shape[1:]))
            frames = torch.cat([frames, padding], dim=0)
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
