import glob

from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class GenImageDataset(Dataset):
    def __init__(self, root_dir: str, phase: str = 'train', crop_size=128):
        self.root_dir = root_dir
        self.phase = phase

        self.images_path = glob.glob(self.root_dir + '/*.png')
        self.crop_size = crop_size

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        # if idx ==0:
        #     np.random.seed(42)
        img = Image.open(self.images_path[idx])
        img = np.array(img, dtype=np.float32)
        h, w = img.shape[:2]
        x, y = np.random.randint(0, w - self.crop_size), np.random.randint(0, h - self.crop_size)

        img = img[y:y + self.crop_size, x:x + self.crop_size, :]
        n_mean, n_var = 0, np.random.randint(0, 76)
        noise = np.random.normal(n_mean, n_var, (self.crop_size, self.crop_size, 3))

        # n_mean, n_var = noise.mean(), noise.var()

        img_n = img + noise
        img_n = np.clip(img_n, a_min=0, a_max=255) / 255

        # trg = np.array([n_mean, n_var]).astype(np.float32)
        # trg = np.zeros(75, dtype=np.int64)
        src1 = np.transpose(img, (2, 0, 1)).astype(np.float32)
        src2 = np.transpose(img_n, (2, 0, 1)).astype(np.float32)
        trg = np.array(n_var, dtype=np.int64)
        return src1, src2, trg
