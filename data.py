import torch.nn
import torch
import os
import numpy as np   
from torch.utils.data import Dataset, DataLoader
import transforms
import config
from skimage.filters import sobel
from skimage.transform import rotate
import random
import cv2
import config
class DataFolder(Dataset):
	def __init__(self, imgs, labels, trainable=True):
		super(DataFolder, self).__init__()
		self.img_paths = imgs
		self.label_paths = labels
		self.trainable = trainable
		assert(len(self.img_paths)==len(self.label_paths))
	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, idx):
		img_path = self.img_paths[idx]
		label_path = self.label_paths[idx]

		img = cv2.imread(img_path)
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		#prior = self.mr(img).astype(float) / 255.0
		#edge = sobel(gray)
		img = cv2.resize(img, config.IMG_SIZE) / 255.0

		label = cv2.imread(label_path, 0)
		if self.trainable:
			label = cv2.resize(label, config.LABEL_SIZE) / 255.0
			label = np.clip(label, 0, 1)
			if random.random() < 0.5:
				img = cv2.flip(img, 1)
				label = cv2.flip(label, 1)
				#prior = cv2.flip(prior, 1)
				#edge = cv2.flip(edge, 1)
			angle = random.choice([0, 90, 180, 270])
			img = rotate(img, angle, clip=True)
			label = rotate(label, angle, clip=True)
			#edge = rotate(edge, angle, clip=True)
			#prior = rotate(prior, angle, clip=True)


		else:
			label = cv2.resize(label, config.LABEL_SIZE) / 255.0
			label = np.clip(label, 0, 1)
		label[label < 0.5] = 0
		label[label > 0.5] = 1
		s = np.sum(label) / np.prod(config.LABEL_SIZE)
		weight = np.zeros_like(label)
		weight[label == 0] = 1. - s
		weight[label == 1] = s
		img = np.transpose(img, [2, 0, 1])
		##float tensor
		img = torch.FloatTensor(img)
		label = torch.FloatTensor(label.astype(np.int))
		weight = torch.FloatTensor(weight)
		return img, label, weight





if __name__ == "__main__":
	FILES = os.listdir(config.DATA_DIR)
	FILES = map(lambda x: os.path.join(config.DATA_DIR, x), FILES)

	IMGS = sorted(filter(lambda x: ".jpg" in x, FILES))
	LABELS = sorted(filter(lambda x: ".png" in x, FILES))
	train_folder = DataFolder(IMGS, LABELS, True)
	img, label, weight = train_folder[0]
	cv2.imshow("img", img)
	cv2.imshow("label", label)
	cv2.waitKey(0)




