#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
# Created on 2025/11/26
# Udated  on 22025/11/26
# @author: Flavio Lichtenstein
# @local: Bioinformatics: CENTD/Molecular Biology; Instituto Butatan


import os # math, sys, pickle, shutil
from os.path import join as osjoin
from tqdm import tqdm
from os.path import exists as exists
from typing import Tuple, Any, List # Optional, Iterable, Set,

# import scipy
# from   scipy.stats import hypergeom

import numpy as np
# import time, json
# from datetime import datetime
# import pandas as pd

# from PIL import Image as PILImage

# import re
# _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

# import seaborn as sns

import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

import torch
from torch import autocast
from torch.amp import grad_scaler
# from torch.optim import Adam
from torch.utils.data import Dataset

# from torch.utils.data import DataLoader
from torchvision import transforms # models, datasets

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

from monai.transforms.io.array import LoadImage
from monai.transforms.utility.array import EnsureChannelFirst, ToTensor
from monai.transforms.spatial.array import Resize, RandFlip, RandRotate, RandZoom
from monai.transforms.intensity.array import ScaleIntensity, RandGaussianNoise # , RandShiftIntensity


from monai.data.dataloader import DataLoader as MonaiDataLoader # CacheDataset
# https://monai-dev.readthedocs.io/en/fixes-sphinx/networks.html
# from monai.networks.nets import DenseNet121
# from monai.losses import DiceCELoss
from monai.utils.misc import set_determinism

# see https://monai.readthedocs.io/en/1.3.0/_modules/monai/networks/nets/efficientnet.html
from monai.networks.nets.efficientnet import EfficientNetBN

from libs.Basic import *

class MyNN(object):
	'''
		my neural network class
	'''
	def __init__(self, crop_or_segment:str, ncrop:int, sel_probes:List, classes:List, 
			     root0_data:str, n_determinism:int=-1, verbose:bool=False):

		if verbose:
			print(">>> PyTorch version", torch.__version__)

			if torch.cuda.is_available():
				print(">> current_device:", torch.cuda.current_device())
				print(">> Device:", torch.cuda.get_device_name(0))
				print(">> CUDA:", torch.version.cuda)
			else:
				print(">>> only CPU")
				raise Exception("\n\n--------------stop ------------------")

		self.root0_data = root0_data
		if not exists(root0_data):
			print(f"Error: there is no root folder: '{root0_data}'")
			raise Exception("--------------stop ------------------")

		#-------- data/samples from HCS or other sources---------------
		self.root_samples = create_dir(self.root0_data, 'samples')
		#------------ we create these data -----------------
		self.root_table   = create_dir(root0_data, 'tables')
		self.root_train_test   = create_dir(root0_data, 'train_and_test')
		self.root_train  = create_dir(self.root_train_test, 'train')
		self.root_test   = create_dir(self.root_train_test, 'test')
		self.root_crop	  = create_dir(root0_data, 'crop')
		self.root_segment = create_dir(root0_data, 'segment')

		if n_determinism > 0:
			set_determinism(seed=n_determinism)

		self.crop_or_segment = crop_or_segment
		self.ncrop = ncrop
		self.sel_probes = sel_probes

		if crop_or_segment == 'crop':
			if sel_probes == []:
				self.fname_model0 = "cell_EfficientNet_B3_ncrop_%d.pt"
			else:
				self.fname_model0 = "cell_EfficientNet_B3_ncrop_%d_probes_%s.pt"
		else:
			if sel_probes == []:
				self.fname_model0 = "cell_EfficientNet_B3_segment_%s.pt"
			else:				
				self.fname_model0 = "cell_EfficientNet_B3_segment_%s_probes_%s.pt"

		self.classes = list(np.unique(classes))
		self.class_to_index = {_class: i for i, _class in enumerate(classes)}
		self.num_classes = len(self.classes)

		self.model = EfficientNetBN(
			model_name="efficientnet-b3",
			spatial_dims=2,
			in_channels=3, 
			num_classes=3,
			pretrained=True   # USE pretrained = better accuracy
		).cuda()

		self.criterion = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.AdamW(self.model.parameters())

		if verbose:
			print(f">>> There are {self.num_classes} classes: {'_'.join(self.classes)}")

		self.scaler = grad_scaler.GradScaler()
		
	def create_monai_EfficientNetBN_b3(self, lr:float=1e-4, weight_decay:float=1e-4,
									   label_smoothing:float=0.1, pretrained:bool=True):
		
		# https://monai.readthedocs.io/en/1.3.0/_modules/monai/networks/nets/efficientnet.html
		'''
			__all__ = [
				"EfficientNet",
				"EfficientNetBN",
				"get_efficientnet_image_size",
				"drop_connect",
				"EfficientNetBNFeatures",
				"BlockArgs",
				"EfficientNetEncoder",
			]
		'''
		self.model = EfficientNetBN(
			model_name="efficientnet-b3",
			spatial_dims=2,
			in_channels=3, 
			num_classes=self.num_classes,
			pretrained=pretrained   # USE pretrained = better accuracy
		).cuda()
	
		# use Label Smoothing
		self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
		# Try AdamW instead of Adam:
		# optimizer = Adam(model.parameters(), lr=lr)
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

		# reset losses and accuracy lists
		self.train_losses = []
		self.test_losses = []
		self.accu_list = []


	def get_model_name(self) -> Tuple[str, str]:
		if self.sel_probes == []:
			fname_model = self.fname_model0%(self.ncrop)
		else:
			fname_model = self.fname_model0%(self.ncrop, ';'.join(self.sel_probes))
		
		filename = osjoin(self.root_samples, fname_model)
		return filename, fname_model

	def save_model(self, verbose:bool=False) -> bool:

		# filename, fname_model
		filename, _ = self.get_model_name()

		try:
			torch.save({
				"model_state": self.model.state_dict(),
				"optimizer_state": self.optimizer.state_dict(),
				"classes": self.classes,
				"class_to_index": self.class_to_index,
				"train_losses": self.train_losses,
				"test_losses": self.test_losses,
				"accu_list": self.accu_list,
				"n": len(self.accu_list),
				"train_loader": self.train_loader,
				"test_loader": self.test_loader,
			},  filename)
			if verbose: print(f"File saved at '{filename}'")
			ret = True
		except:
			print(f"Error: could not save '{filename}'")
			ret = False

		return ret

	def read_model(self, verbose:bool=False) -> Tuple[bool, Tuple[Any, List, List, List]]:
		# filename, fname_model = self.get_model_name()
		filename, _ = self.get_model_name()

		if not exists(filename):
			print("Do not change train and test dataloaders...")
			return False, (self.model, [], [], [])
		
		try:
			model = self.model
			optimizer = self.optimizer

			checkpoint = torch.load(filename, map_location="cuda", weights_only=False)

			model.load_state_dict(checkpoint["model_state"])
			optimizer.load_state_dict(checkpoint["optimizer_state"])

			train_losses = checkpoint["train_losses"]
			test_losses = checkpoint["test_losses"]
			accu_list = checkpoint["accu_list"]
			train_loader = checkpoint["train_loader"]
			test_loader  = checkpoint["test_loader"]

			if verbose: print(f"Loading model with {len(train_losses)} epochs from '{filename}'")
		except:
			print(f"Error: could not read '{filename}'")
			return False, (self.model, [], [], [])

		self.train_losses = train_losses
		self.test_losses = test_losses
		self.accu_list = accu_list

		self.model = model
		self.optimizer = optimizer

		print("Reloading train and test dataloaders...")
		self.train_loader = eval(train_loader) if isinstance(train_loader, str) else train_loader
		self.test_loader  = eval(test_loader)  if isinstance(test_loader,  str) else test_loader

		return True, (model, train_losses, test_losses, accu_list)

	def set_train_and_test_dataset(self, ds_train:Dataset, ds_test:Dataset, 
								   batch_size:int=16, shuffle:bool=True, num_workers:int=4):
		self.ds_train = ds_train
		self.ds_test  = ds_test

		train_loader = MonaiDataLoader(ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
		test_loader  = MonaiDataLoader(ds_test,  batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

		self.train_loader = train_loader
		self.test_loader  = test_loader
	
	def set_train_and_test_dataloader(self, train_loader, test_loader):
		self.train_loader = train_loader
		self.test_loader  = test_loader

	def train_monai_model(self, n_epochs:int=10, n_max_repeat:int=5, verbose:bool=False):

		n_count_repeat = 0
		
		maxi = 0.0 if self.accu_list== [] else max(self.accu_list)
		self.maxi = maxi

		for epoch in range(n_epochs):
			self.model.train()
			
			tot_train_loss = 0.0
			for imgs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
				imgs = imgs.cuda()
				labels = labels.cuda()

				self.optimizer.zero_grad()

				# with autocast(device_type="cuda", enabled=True, dtype=torch.float16):
				with autocast('cuda', dtype=torch.float16):
					outputs = self.model(imgs)
					train_loss = self.criterion(outputs, labels)
				
				self.scaler.scale(train_loss).backward()
				self.scaler.step(self.optimizer)
				self.scaler.update()
				
				tot_train_loss += train_loss.item()

			tot_test_loss = 0.0
			tot_accuracy = 0.0
				
			with torch.no_grad(), autocast('cuda', dtype=torch.float16):
				for imgs, labels in self.test_loader:
					imgs = imgs.cuda()
					labels = labels.cuda()
			
					outputs = self.model(imgs)
					test_loss = self.criterion(outputs, labels)
			
					tot_test_loss += test_loss.item()
			
					best_pred = outputs.argmax(1)
					# Because .mean() gives the accuracy for the batch
					tot_accuracy += (best_pred == labels).float().mean().item()

			train_loss = tot_train_loss / len(self.train_loader)
			test_loss  = tot_test_loss  / len(self.test_loader)
			accuracy   = tot_accuracy   / len(self.test_loader)
			
			self.train_losses.append(train_loss)
			self.test_losses.append(test_loss)
			self.accu_list.append(accuracy)

			if accuracy > maxi:
				n_count_repeat = 0
				maxi = accuracy
				if verbose: print(f">> new best accuracy = {100*accuracy:.1f}")

				self.save_model(verbose=verbose)
			else:
				n_count_repeat +=1
				if n_count_repeat >= n_max_repeat:
					print("Warning: did not improve, breaking the training loop")
					break
				
			print(f"Epoch {epoch+1} - accuracy: {accuracy*100:.1f}% loss train: {train_loss:.4f} test: {test_loss:.4f}")

		self.maxi = maxi
		self.save_model(verbose=verbose)

		print("-------- training complete ----------")


	def plot_losses_and_accuracy(self, figsize:Tuple=(12,5)):
		epochs_range = range(1, len(self.train_losses) + 1)
		plt.figure(figsize=figsize)
		
		#------ loss plot ------------------
		plt.subplot(1, 2, 1)
		plt.plot(epochs_range, self.train_losses, marker='o', label='Train Loss')
		plt.plot(epochs_range, self.test_losses, marker='o', label='Test Loss')
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		plt.grid(True)
		plt.title("Training & Test Losses")
		
		#----- accuracy plot ---------------
		plt.subplot(1, 2, 2)
		plt.plot(epochs_range, self.accu_list, marker='o', label='Test Accuracy')
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.title("Test Accuracy")
		plt.legend()
		plt.grid(True)
		
		plt.tight_layout()
		plt.show()

	def evaluate_loss_accuracy(self, model, loader):
		model.eval()
		tot_accuracy = 0
		loss_sum = 0

		with torch.no_grad():
			for images, labels in loader:
				images = images.cuda()
				labels = labels.cuda()
			
				with autocast(device_type='cuda', dtype=torch.float16):
					outputs = model(images)
					loss = self.criterion(outputs, labels)
					preds = outputs.argmax(1)

				loss_sum += loss.item()
				tot_accuracy += (preds == labels).mean().item()

		return loss_sum / len(loader), tot_accuracy / len(loader)
	
	class CellDataset_b3(Dataset):

		def __init__(self, items):
			self.items = items
			self.train_transforms = self.create_data_transforms_b3()

		def __len__(self):
			return len(self.items)

		def __getitem__(self, idx):
			return self.apply_transforms(self.items[idx])
		
		def apply_transforms(self, sample):

			img = sample["img"]
			label = sample["label"]

			for train_transf_method in self.train_transforms:
				img = train_transf_method(img)

			return img, torch.tensor(label, dtype=torch.long)		
		
		def create_data_transforms(self, size_x:int, size_y:int) -> Tuple[Any, Any]	:
			self.train_transform = transforms.Compose([
				transforms.Resize(size=(size_x, size_y)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])

			self.test_transform = transforms.Compose([
				transforms.Resize(size=(size_x, size_y)),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

			return self.train_transform, self.test_transform
		

		def create_data_transforms_b3(self) -> list:

			train_transforms = [
				LoadImage(image_only=True),
				EnsureChannelFirst(),
				Resize((300, 300)),
				RandFlip(prob=0.5),
				RandRotate(range_x=0.2, prob=0.5),
				RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
				RandGaussianNoise(prob=0.2),
				ScaleIntensity(),
				ToTensor()
			]
			
			return train_transforms
				
					