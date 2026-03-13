#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
# Created on 2025/01/26
# Udated  on 2025/11/26
# @author: Flavio Lichtenstein
# @local: Bioinformatics: CENTD/Molecular Biology; Instituto Butatan


import os, yaml, shutil # sys, math, pickle
from os.path import join as osjoin
from os.path import exists as exists
from typing import Tuple, Any, List # Optional, Iterable, Set,

# import scipy
# from   scipy.stats import hypergeom

import numpy as np
# import time, json
from datetime import datetime
import pandas as pd
# from sklearn.utils import shuffle

# import PIL
from PIL import Image as PILImage

import re
# _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

# import seaborn as sns

import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

import cv2
# from natsort import natsorted
# from tqdm import trange

# from IPython.display import Markdown

# import plotly.graph_objects as go
# import plotly.express as px

from libs.Basic import *
from libs.parallel_image_lib import *

from cellpose import models, core, io, plot
import torch


class Cellpose(object):
	'''
		Cellpose
			- Python 3.8.19
			- cellpose_env
	'''
	def __init__(self, root0_data:str='.', verbose:bool=False):

		if core.use_gpu()==False:
			print(ImportError("No GPU access, change your runtime"))

		if verbose:
			print("torch:", torch.__version__)
			print("\n------- logger_setup() -----------")
			print(io.logger_setup())

			if core.use_gpu()==True:
				print("\n------- GPU Ok -----------")
				print(">> current_device:", torch.cuda.current_device())
				print(">> Device:", torch.cuda.get_device_name(0))
				print(">> CUDA:", torch.version.cuda, '\n')
				print(">> torch.version.cuda:", torch.version.cuda)

		self.root0_data = root0_data
		if not exists(root0_data):
			print(f"Error: there is no root folder: '{root0_data}'")
			raise Exception("--------------stop ------------------")


		#-------- data/samples from HCS or other sources---------------
		self.root_samples = create_dir(self.root0_data, 'samples')

		self.root_train_test = create_dir(root0_data, 'train_and_test')
		self.root_table   = create_dir(root0_data, 'tables')
		self.root_train   = create_dir(self.root_train_test, 'train')
		self.root_test    = create_dir(self.root_train_test, 'test')
		self.root_crop	  = create_dir(root0_data, 'crop')
		self.root_segment = create_dir(root0_data, 'segment')
		
		self.model_name = ''
		self.class_names = []

		self.probe = ''

		self.dir_origins = []
		self.dir_names = []
		
		self.crop_segment = ''
		self.ncrop = 5

		self.sel_probes = []
		self.perc = .6
		self.classes = []
		self.class_to_index = {}
		
		self.plate = ''
		self.experiment = ''

		if verbose:
			print(f"\nPlease set plate and experiment using create_roots(plate, experiment)\n")

	def set_default_parameters(self, root_yaml:str, verbose:bool=False):

		filename_yaml = osjoin(root_yaml, 'params.yml')
		with open(filename_yaml, 'r') as file:
			dic_yml = yaml.safe_load(file)

		self.dic_yml = dic_yml

		self.model_name = dic_yml['model_name']
		self.crop_segment = dic_yml['crop_segment']    
		self.ncrop = 5
		self.dic_plate = dic_yml['dic_plate']

		self.probes = []
		self.dir_names = []
		self.dir_origins = []
		self.probes = []

		if verbose:
			print(">> model_name  ", self.model_name)
			print(">> crop_segment", self.crop_segment)
			print(">> dic_plate   ", len(self.dic_plate))
			print(">> ncrop   ", self.ncrop)


	def set_plate_params(self, plate:str, verbose:bool=False):

		self.create_roots_plate(plate, verbose=verbose)

		try:
			dic = self.dic_plate[plate]
			
			self.class_names = dic['class_names']
			self.dir_names = dic['dir_names']
			self.dir_origins0 = dic['dir_origins']
			self.probes = dic['probes']

			self.dir_origins = []

			if verbose:
				print("")
				print("\tclass_names", '\t', self.class_names)
				print("\tprobes", '\t', self.probes)
				print("\tdir_names", '\t', self.dir_names)
				print("\tdir_origins", '\t', self.dir_origins)
		except:
			self.class_names = []
			self.dir_names = []
			self.dir_origins0 = []
			self.dir_origins = []
			self.probes = []
			self.dir_names = []
			self.probe = None

			print(f"Error: please configure the plate {plate}")


	def create_roots_plate(self, plate:str, verbose:bool=False):

		self.plate = plate
		self.experiment = ''

		self.root_plate	    = osjoin(self.root_samples, plate)

		self.root_tbl_plate  = create_dir(self.root_table, plate)
		self.root_crop_plate = create_dir(self.root_crop, plate)
		self.root_seg_plate  = create_dir(self.root_segment, plate)

		self.experiments = self.list_experiments(flg_is_dir=True)

		if verbose:
			print(">>> plate", plate, 'experiment must be set')
			print(f"\troot_plate:	  '{self.root_plate}'")
			print(f"\troot_tbl_plate:  '{self.root_tbl_plate}'")
			print(f"\troot_crop_plate: '{self.root_crop_plate}'")
			print(f"\troot_seg_plate:  '{self.root_seg_plate}'")

	def list_plates(self, s_start:str='Plate') -> list:
		plates = [x for x in os.listdir(self.root_samples) if x.startswith(s_start)]
		return plates

	def create_roots_experiment(self, experiment:str, verbose:bool=False):

		self.experiment = experiment
		mat = experiment.split(' ')
		self.probe = mat[0]

		if self.probe not in self.probes:
			print(f"Error: create_roots_experiment() -> probe '{self.probe}' not in probes list: {self.probes}")
			self.dir_origins = []
		else:
			self.dir_origins = [x%(self.probe) for x in self.dir_origins0]

		self.root_image	     = create_dir(self.root_plate, experiment)
		self.root_crop_image = create_dir(self.root_crop_plate, experiment)
		self.root_seg_image  = create_dir(self.root_seg_plate, experiment)

		self.root_tbl_experiment = create_dir(self.root_tbl_plate, experiment)
		self.root_tbl_image	  = self.root_tbl_experiment


		if verbose:
			print(f">>> experiment '{experiment}' - plate {self.plate} and probe '{self.probe}'")
			print(f"\troot_tbl img/exp: '{self.root_tbl_experiment}'")
			print(f"\troot image:	   '{self.root_image}'")
			print(f"\troot crop image:  '{self.root_crop_plate}'")
			print(f"\troot seg image:   '{self.root_seg_image}'")

			print(f"\n\nplate: '{self.plate}' experiment: '{self.experiment}' probe: '{self.probe}' -> {'is ok' if self.probe in self.probes else 'error'}")

	def list_experiments(self, flg_is_dir:bool=True, verbose:bool=False) -> list:
		if flg_is_dir:
			exps = [x for x in os.listdir(self.root_plate) if os.path.isdir(os.path.join(self.root_plate, x))]
		else:
			exps = os.listdir(self.root_plate)

		if verbose: print(f">>> There are {len(exps)}_experiments() in '{self.root_plate}'")

		self.get_probes(exps, verbose=verbose)

		return exps
	
	def get_probes(self, exps:List, verbose:bool=False) -> List:
		probes = []
		for exp in exps:
			mat = exp.split(' ')
			probe = mat[0]
			if probe not in probes:
				probes.append(probe)

		self.probes = probes
		if verbose: print(f">>> {self.plate} has probes:", probes)
		return probes


	def create_roots(self, plate:str, experiment:str):
		self.create_roots_plate(plate)
		self.create_roots_experiment(experiment)

	def set_data_origin_and_create_roots_to_train_and_test(self, image_example:PILImage.Image, 
							verbose:bool=False) -> Tuple[str, dict, dict]:

		width, height = image_example.size
		size_x = int(width/self.ncrop)
		size_y = int(height/self.ncrop)

		self.width = width
		self.height = height
		self.size_x = size_x
		self.size_y = size_y

		if self.crop_segment == 'crop':
			if verbose: print(width, height, '->', size_x, size_y)
			fname0 = f'best_model_weights_for_{self.model_name}_probe_{self.probe}_{self.plate}_crop_{self.ncrop}'
		else:
			fname0 = f'best_model_weights_for_{self.model_name}_probe_{self.probe}_{self.plate}_segment_each_cell'

		self.model_fname = fname0 + '.pt'
		self.model_table = fname0 + '.tsv'

		self.root_data = self.root_seg_plate if self.crop_segment == 'segment' else self.root_crop_plate

		if verbose: print(f">>>Is a {self.crop_segment} simulation where root_data is '{self.root_data}'")

		dic_root_train, dic_root_test, dic_root_ori = {}, {}, {}

		for i, class_name in enumerate(self.class_names):

			dir_name = self.dir_names[i]
			dir_ori  = self.dir_origins[i]

			dic_root_train[class_name] = create_dir(self.root_train, dir_name)
			dic_root_test[class_name]  = create_dir(self.root_test, dir_name)
			dic_root_ori[class_name]   = create_dir(self.root_data, dir_ori)
		
		self.dic_root_train = dic_root_train
		self.dic_root_test = dic_root_test
		self.dic_root_ori = dic_root_ori

		if verbose:
			if self.crop_segment == 'segment':
				print("Segment at:", self.root_data)
			else:
				print("Crop at:", self.root_data)

			print("Origin:")
			for key, val in self.dic_root_ori.items():
				print(f"\t{key:12} -> {val}")

			print("Train:")
			for key, val in self.dic_root_train.items():
				print(f"\t{key:12} -> {val}")

			print("Test:")
			for key, val in self.dic_root_test.items():
				print(f"\t{key:12} -> {val}")

			print("")
		
		return self.root_data, dic_root_train, dic_root_test


	def clean_train_and_test(self, verbose:bool=False) -> bool:
		try:
			shutil.rmtree(self.root_train_test)
		except:
			pass

		try:
			self.root_train_test = create_dir(self.root0_data, 'train_and_test')
			self.root_train  = create_dir(self.root_train_test, 'train')
			self.root_test   = create_dir(self.root_train_test, 'test')
			ret = True
		except:
			print("Impossible to create", self.root_train_test)
			ret = False

		if verbose and ret:
			print(f"root_train_test: '{self.root_train_test}'")
			print(f"train:           '{self.root_train}'")
			print(f"test:            '{self.root_test}'")

		return ret

	def copy_data_train_test(self, max_images:int=1200, 
						    perc_train:float=.60, ncrop:int=5, image_type:str='png', 
							verbose:bool=False) -> Tuple[bool, pd.DataFrame]:

		if not self.clean_train_and_test(verbose=verbose):
			return False, pd.DataFrame()
		
		s_end = f'_ncrop_{ncrop}.{image_type}'
		
		dic={}; icount=-1
		for i, class_name in enumerate(self.class_names):

			dir_origin = self.dir_origins[i]
			dir_target = self.dir_names[i]

			# root_data is crop or segment
			root_data_case = os.path.join(self.root_data, dir_origin)
			root_target_train = create_dir(self.root_train, dir_target)
			root_target_test  = create_dir(self.root_test,  dir_target)
		
			if self.crop_segment == 'crop':
				fnames = [x for x in os.listdir(root_data_case) if x.endswith(s_end)]
			else:
				print(">>> segment case - create a method to list images")
				raise Exception("\n\n--------------stop ------------------")
			
			n = len(fnames)
			if n < max_images:
				maxi = n
			else:
				maxi = max_images
		
			random.shuffle(fnames)
			samples = fnames[:maxi]
			n_samples = len(samples)
		
			n_train_samples = int(n_samples*perc_train)
			train_samples = samples[:n_train_samples]
			test_samples  = samples[n_train_samples:]
		
			n_test_samples  = len(test_samples)
		
			icount += 1
			
			dic[icount] = {}
			dic2 = dic[icount]
			
			dic2['class_name'] = class_name
			dic2['n'] = n
			dic2['perc_train'] = perc_train
			dic2['n_samples'] = n_samples
			dic2['n_train_samples'] = n_train_samples
			dic2['n_test_samples'] = n_test_samples
			dic2['root_data'] = root_data_case
			dic2['root_target_train'] = root_target_train
			dic2['root_target_test'] = root_target_test
			dic2['train_samples'] = train_samples
			dic2['test_samples'] = test_samples
			
			print(f"{i}) {class_name:12} n={n:5} -> {root_data_case}")

		df = pd.DataFrame(dic).T

		for i, row in df.iterrows():
			class_name = row.class_name
			root_data = row.root_data

			#------------ train data -------------------------------
			root_target_train = row.root_target_train
			train_samples = row.train_samples
			train_samples = train_samples if isinstance(train_samples, list) else eval(train_samples)
			
			if verbose:
				print(f">>> moving class '{class_name}': {len(train_samples)} train samples from \n{root_data} to {root_target_train}")

			for fname in train_samples:
				filename = os.path.join(root_data, fname)
				try:
					shutil.copy(filename, root_target_train)
				except:
					print(f"Error: moving {filename} to {root_target_train}")
					return False, df
						
			#------------ test data --------------------------------
			root_target_test  = row.root_target_test
			test_samples = row.test_samples
			test_samples = test_samples if isinstance(test_samples, list) else eval(test_samples)
			
			if verbose:
				print(f">>> moving class '{class_name}': {len(test_samples)} test samples from \n{root_data} to {root_target_test}")
	
			for fname in test_samples:
				filename = os.path.join(root_data, fname)
				try:
					shutil.copy(filename, root_target_test)
				except:
					print(f"Error: moving {filename} to {root_target_test}")
					return False, df

		return True, df

	def create_train_and_test_dataset(self, ncrop:int=5, sel_probes:list=[],
								      perc_train:float=0.4, perc_test:float=0.2) -> Tuple[List, List, pd.DataFrame]:
		self.ncrop = ncrop
		self.sel_probes = sel_probes
		self.perc_train = perc_train
		self.perc_test  = perc_test

		if perc_train <= 0 or perc_test <= 0 or perc_train + perc_test > 1.0:
			print("Error: perc_train + perc_test must be <= 1.0")
			return [], [], pd.DataFrame()

		dic = {}
		train_list, test_list = [], []
		plates = self.list_plates(s_start='Plate')

		classes = []
		for plate in plates:
			self.set_plate_params(plate=plate, verbose=False)

			for experiment in self.experiments:
				
				mat = experiment.split(' - ')
				probe = mat[0]
				perturb = mat[1]

				if len(sel_probes) > 0 and probe not in sel_probes:
					continue
					
				classes.append(experiment)

		classes = list(np.unique(classes))
		class_to_index = {_class: i for i, _class in enumerate(classes)}

		self.classes = classes
		self.class_to_index = class_to_index

		icount=-1
		for plate in plates:
			self.set_plate_params(plate=plate, verbose=False)

			for experiment in self.experiments:

				mat = experiment.split(' - ')
				probe = mat[0]
				perturb = mat[1]

				if len(sel_probes) > 0 and probe not in sel_probes:
					continue

				self.create_roots_experiment(experiment, verbose=False)
				fname_imgs = self.list_crop_images_already_set(ncrop=ncrop, image_type='png', verbose=False)
				fname_imgs = np.array(fname_imgs)

				n = len(fname_imgs)
				lista = list(np.arange(0, n))
				random.shuffle(lista)
				n_train = int(n*perc_train)
				n_test  = int(n*perc_test)

				fname_imgs_train = fname_imgs[lista[:n_train]]
				fname_imgs_test  = fname_imgs[lista[-n_test:]]

				key = f"{plate} - {experiment}"
				icount += 1
				dic[icount] = {}
				dic2 = dic[icount]
				dic2['plate_exp'] = key
				dic2['plate'] = plate
				dic2['experiment'] = experiment

				dic2['probe'] = probe
				dic2['perturb'] = perturb
				
				dic2['n'] = len(fname_imgs)
				dic2['n_train'] = len(fname_imgs_train)
				dic2['n_test']  = len(fname_imgs_test)
				dic2['root'] = self.root_crop_image
				
				for fname in fname_imgs_train:
					train_list.append({"img": os.path.join(self.root_crop_image, fname), "label": class_to_index[experiment] })

				for fname in fname_imgs_test:
					test_list.append({"img": os.path.join(self.root_crop_image, fname), "label": class_to_index[experiment] })
					
		df = pd.DataFrame(dic).T
		df = df.sort_values(['probe', 'perturb', 'plate'])
		df = df.reset_index()
		
		return train_list, test_list, df

	def list_images_already_set(self, image_type:str='tif', verbose:bool=False) -> list:
		files = [x for x in os.listdir(self.root_image) if x.endswith(image_type)]
		if verbose: print(f"There are {len(files)} {image_type}s in '{self.root_image}'")
		return files

	def list_crop_images_already_set(self, ncrop:int=5, image_type:str='png', verbose:bool=False) -> list:
		s_end = f'_ncrop_{ncrop}.{image_type}'
		files = [x for x in os.listdir(self.root_crop_image) if x.endswith(s_end)]
		if verbose: print(f"There are {len(files)} {image_type}s in '{self.root_crop_image}'")
		return files


	def read_PIL_image(self, fname:str, root_image:str='', verbose:bool=False) -> PILImage.Image:

		if root_image is None or root_image == '':
			root_image = self.root_image
		
		filefig = osjoin(root_image, fname)
		try:
			image = PILImage.open(filefig)
			if verbose: print(f"PIL image read: shape {image.size} mode {image.mode} - '{filefig}'")
		except:
			print(f"Error: could not read: '{filefig}'")
			self.pil_image = PILImage.new("RGB", (10, 10), (255, 255, 255))
			return self.pil_image
		
		self.pil_image = image
		return image

	def save_PIL_image(self, image, fname_wo_type:str, root_image:str, image_type:str='png', 
					   lossless:bool=True, force:bool=False, verbose:bool=False):

		fname = fname_wo_type + '.' + image_type

		filefig = osjoin(root_image, fname)

		if exists(filefig) and not force:
			return True

		try:
			image.save(filefig, lossless=lossless, format=image_type)
			if verbose: print(f"PIL image saved: shape {image.size} mode {image.mode} - '{filefig}'")
		except:
			print(f"Error: could not save: '{filefig}'")
			return False
		
		return True


	def read_display_img(self, fname:str, figsize:tuple=(8,8), verbose:bool=False) -> Tuple[Any, Any]:
		
		image = self.read_PIL_image(fname=fname, verbose=False)
		if image is None:
			return None, None

		fig = plt.figure(figsize=figsize)
		plt.imshow(image);

		if verbose: 
			print(f"PIL image '{fname}' in '{self.root_image}' has shape: {image.size}, mode {image.mode}.\n")

		return fig, plt
	
	def display_img(self, image:Any=None, figsize:tuple=(8,8), verbose:bool=False) -> Tuple[Any, Any]:
		
		if image is None:
			return None, None

		fig = plt.figure(figsize=figsize)
		plt.imshow(image)

		if verbose:
			print(f"\nImage has shape: {image.size}, mode {image.mode}.\n")

		return fig, plt
	
	def image_properties(self, image:Any=None, verbose:bool=False) -> Tuple[Tuple[int,int], str]:
		size = image.size
		mode = image.mode
		
		if verbose: print(f"\nImage has shape: {size}, mode {mode}.\n")

		return size, mode


	def crop_img(self, left:int, upper:int, right:int, lower:int, verbose:bool=False):

		if self.pil_image is None:
			print("Error: no (PIL) image")
			return None
		
		image = self.pil_image
		width, height = image.size

		del_width = right-left
		del_height = abs(upper-lower)

		if del_width > width:
			print(f"Error: crop width {del_width} > width {width}")
			return None
		
		if del_height > height:
			print(f"Error: crop height {del_height} > height {height}")
			return None

		box = (left, upper, right, lower)

		try:
			image_crop = image.crop(box)
			self.image_crop = image_crop
			if verbose: print(f"Crop: {image.size} with box {box}")
		except:
			print(f"Error: could not crop: {image.size} with box {box}")
			self.image_crop = None
			return None
		
		return image_crop
	
	def crop_and_display_squares(self, fname:str, lossless:bool=True,
								 image_type:str='png', figsize:tuple=(8,8), 
								 force:bool=False, verbose:bool=False) -> Tuple[pd.DataFrame, Any, Any]:

		df = self.crop_squares_already_set(fname, lossless=lossless, 
										   image_type=image_type, force=force, verbose=verbose) 

		if df is None or df.empty:
			return df, None, None
		
		fig, axes = self.display_cropped_img_from_df(df, figsize=figsize)

		return df, fig, axes
			
	def crop_squares_already_set(self, fname:str, ncrop:int=5,
					 lossless:bool=True, image_type:str='png',
					 force:bool=False, verbose:bool=False) -> pd.DataFrame:
		
		if ncrop < 2 or ncrop > 20:
			print(f"Error: ncrop must be between 2 and 20, not {ncrop}")
			return pd.DataFrame()
		
		# fname.png .tif .tiff  .jpeg .jpg
		fname_tsv = fname.split('.')[0] + f'_ncrop_{ncrop}.tsv'
		fname_tsv = title_replace(fname_tsv)
		filename = osjoin(self.root_tbl_experiment, fname_tsv)

		if os.path.exists(filename) and not force:
			df = pdreadcsv(fname_tsv, self.root_tbl_experiment, verbose=verbose)
			return df

		image = self.read_PIL_image(fname=fname, verbose=verbose)

		fname_crop_img0 = self.remove_img_type_from_fname(fname)

		width, height = image.size

		del_width = int(width/ncrop)
		del_height = int(height/ncrop)		

		if image is None:
			return pd.DataFrame()
	
		icount = -1; dic={}
		for ix in range(ncrop):
			xmin = ix*del_width

			if ix == ncrop-1:
				xmax = width
			else:
				xmax = xmin + del_width
			
			for iy in range(ncrop):
				ymin = iy*del_height
				
				if iy == ncrop-1:
					ymax = height
				else:
					ymax = ymin + del_height

				# box = (left, upper, right, lower)
				box = (xmin, ymin, xmax, ymax)
				crop_img = image.crop(box)

				icount +=1
				dic[icount] = {}
				dic2 = dic[icount]

				dic2['xmin'] = xmin
				dic2['xmax'] = xmax
				dic2['ymin'] = ymin
				dic2['ymax'] = ymax

				dic2['del_x'] = xmax-xmin
				dic2['del_y'] = ymax-ymin

				fname_wo_type = fname_crop_img0 + f"_crop_{icount}_ncrop_{ncrop}"
				dic2['icount'] = icount
				dic2['ncrop'] = ncrop
				dic2['fname'] = fname_wo_type + '.' + image_type
				dic2['image_type'] = image_type
				
				ret = self.save_PIL_image(crop_img, fname_wo_type=fname_wo_type, root_image=self.root_crop_image, image_type=image_type, lossless=lossless, force=False, verbose=verbose)
				if not ret:
					raise Exception("------------- stop: saving error ---------------")

		df = pd.DataFrame(dic).T
		_ = pdwritecsv(df, fname_tsv, self.root_tbl_experiment, verbose=verbose)
  
		return df


	def remove_img_type_from_fname(self, fname:str) -> str:
		if fname is None or fname == '':
			print(f"Error: remove_img_type_from_fname() - '{fname}'")
			return fname

		img_types = ['.png', '.tif', '.gif', '.jpeg', '.jpg', '.bmp']

		for img_type in img_types:
			n = len(img_type)

			if fname[:-n].lower() == img_type:
				fname = fname[:-n]
				break

		return title_replace(fname)
	
	def replace_img_type_from_fname(self, fname:str, to_type:str='png') -> str:
		img_types = ['.png', '.tif', '.gif', '.jpg', '.bmp']

		for img_type in img_types:
			n = len(img_type)

			if fname[:-n].lower() == img_type:
				return fname[:-n] + '.' + to_type

		return fname + '.' + to_type


	def display_cropped_img_from_df(self, df, figsize:tuple=(8,8), transpose:bool=False,
									margin:tuple=(0.1, 0.1, 0.9, 0.9, 0.05, 0.05) ) -> Tuple[Any, Any]:
		
		fig, axes = plt.subplots(self.ncrop, self.ncrop, figsize=figsize)

		i=-1
		for ix in range(self.ncrop):
			for iy in range(self.ncrop):
				i += 1
				row = df.iloc[i]

				fname = row['fname']

				crop_image = self.read_PIL_image(fname, self.root_crop_image)
				if crop_image is None:
					continue

				if transpose:
					ax = axes[ix][iy]
				else:
					ax = axes[iy][ix]
			
				ax.imshow(crop_image)
				ax.xaxis.set_visible(False)
				ax.yaxis.set_visible(False)

		# left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05
		try:
			left, bottom, right, top, wspace, hspace = margin
			plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
		except:
			print(f"Error: margin = {margin}")
		
		plt.tight_layout()

		return fig, axes
	
	def show_images(self, images:List, labels:List, preds:List, 
				    fontsize:int=14, figsize:tuple=(12, 20)):
		
		plt.figure(figsize=figsize)

		for i, image in enumerate(images):
			
			plt.subplot(5, 2, i + 1, xticks=[], yticks=[])
			
			image = image.numpy().transpose((1, 2, 0))
			
			means = np.array([0.485, 0.456, 0.406])
			stds  = np.array([0.229, 0.224, 0.225])
			
			image = image * stds + means
			image = np.clip(image, 0., 1.)
			plt.imshow(image)

			col = 'red' if (preds[i] != labels[i]) else 'navy'
				
			plt.xlabel(f'orig {self.class_names[int(labels[i].numpy())]}', color=col, fontsize=fontsize)
			plt.ylabel(f'pred {self.class_names[int(preds[i].numpy())]}',  color=col, fontsize=fontsize)
		
		plt.tight_layout()
		plt.show()	
	
class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, cp:Cellpose, df:pd.DataFrame, transform, train_or_test:str='train'):
        
        self.transform = transform

        self.class_names = self.class_names
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples = []
        for _, row in df.iterrows():

            class_name = row.class_name

            label = self.class_to_idx[class_name]

            if train_or_test == 'train':
                samples = row.train_samples
                root = row.root_target_train
            elif train_or_test == 'test':
                samples = row.test_samples
                root = row.root_target_test
            else:
                print("Error: train or test?")
                raise ValueError('\n\n--------------- stop --------------\n')

            samples = samples if isinstance(samples, list) else eval(samples)

            self.samples += [ (img_name, root, label) for img_name in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_name, root, label = self.samples[index]

        image = self.read_PIL_image(img_name, root).convert("RGB")
        x = self.transform(image)

        return x, label

	
class Image(object):
	def __init__(self, root0_data:str, root_img:str,
				 deltax:int=250, deltay:int=250, 
				 image_size_x:int=2048, image_size_y:int=2048):

		self.root0_data = root0_data
		self.root_img = create_dir(root0_data, root_img)

		root_save_img = root_img.strip()+'_segments'
		self.root_save_img  = create_dir(root0_data, root_save_img)

		self.image_size_x = image_size_x
		self.image_size_y = image_size_y

		self.maxi_x0 = image_size_x
		self.maxi_y0 = image_size_y

		self.maxi_x0m1 = image_size_x-1
		self.maxi_y0m1 = image_size_y-1

		self.deltax = deltax
		self.deltay = deltay

		filelog = 'image.log'
		self.set_filelog(filelog)

	def read_PIL_image(self, fname_img:str, verbose:bool=False) -> PILImage.Image:
		fileimg = osjoin(self.root_img, fname_img)
		
		try:
			image = PILImage.open(fileimg)
			if verbose: print(f"PIL image read: shape {image.size} mode {image.mode} - '{fileimg}'")
		except:
			print(f"Error: could not read: '{fileimg}'")
			self.pil_image = PILImage.new("RGB", (100, 10), (255, 255, 255))
			return self.pil_image

		self.img = image
		return image


	def read_segmented_img(self, fname_img:str, verbose:bool=False):
		fileimg = osjoin(self.root_save_img, fname_img)
		
		try:
			if verbose: print(f"reading segmented: '{fileimg}'")
			img = plt.imread(fileimg)
		except:
			print(f"could not read: {'fileimg'}")
			img = None

		self.img = img
		return img


	def display_img(self, img:PILImage.Image, cmap:str='gray', figsize:tuple=(8,8),
					left:int=0, bottom:int=0, right:int=1, top:int=1, wspace:int=0, hspace:int=0):
		fig = plt.figure(figsize=figsize)
		ax = fig.add_subplot(111)

		ax.imshow(img,cmap)

		# ax.set_xticks([])
		# ax.set_yticks([])

		# ax.set_xticklabels([])
		# ax.set_yticklabels([])

		plt.axis('off')
		plt.tight_layout()
		plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
		plt.show()

		return fig, ax


	def convert_img_to_gray(self, img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	def reduce_to_3_gray_patterns(self, img, Ninf:int, Nmid_val:int, Nmid:int, Nmax:int) -> object:
		return np.array([ [0 if x < Ninf else Nmid_val if x < Nmid else Nmax for x in seq] for seq in img])

	
	def display_all_contours(self, img:PILImage.Image, cmap:str='gray', figsize:tuple=(12,8)):
		contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_CCOMP, method=scv2.CHAIN_APPROX_SIMPLE)

		external_contours = np.zeros(img.size)

		for i in range(len(contours)): # 
			
			# last column in the array is -1 if an external contour (no contours inside of it)
			if hierarchy[0][i][3] == -1:
				
				# We can now draw the external contours from the list of contours
				cv2.drawContours(image=external_contours, contours=contours, contourIdx=i, color=255, thickness=-1)

		fig, ax = self.display_img(img=external_contours, cmap=cmap, figsize=figsize)
		return fig, ax, contours, hierarchy


	def define_xy_min_max(self, contours_i):
		# max has +1
		x_min, y_min, delx, dely = cv2.boundingRect(contours_i)

		return x_min, x_min+delx, y_min, y_min+dely


	def define_xy_min_max_loop(self, contours_i):
		
		x_min, x_max, y_min, y_max = self.maxi_x0, 0, self.maxi_y0, 0

		for k in range(len(contours_i)):
			x, y = contours_i[k][0]
			
			if x < x_min:
				x_min = x
			if x > x_max:
				x_max = x
		
			if y < y_min:
				y_min = y
			if y > y_max:
				y_max = y
		
		return x_min, x_max, y_min, y_max


	def define_quad(self, contours_i):
		x_min, x_max, y_min, y_max = self.maxi_x0m1, 0, self.maxi_y0m1, 0
		
		for k in range(len(contours_i)):
			# for a gray scale image
			x, y = contours_i[k][0]
			
			if x < x_min:
				x_min = x
			if x > x_max:
				x_max = x
		
			if y < y_min:
				y_min = y
			if y > y_max:
				y_max = y
		
		return [ [x_min,y_min], [x_max, y_min], [x_min, y_max], [x_max,y_max]]

	def calc_area(self, y_min, y_max, x_min, x_max):
		area = (x_max-x_min) * (y_max-y_min)
		return area if area >= 0 else -area

	def calc_2_overlaps(self, coord1, coord2, verbose:bool=False):

		y_min1, y_max1, x_min1, x_max1 = coord1
		y_min2, y_max2, x_min2, x_max2 = coord2

		if verbose:
			print(y_min1, y_max1, x_min1, x_max1)
			print(y_min2, y_max2, x_min2, x_max2)
		
		if y_max2 < y_min1:
			if verbose: print('before y')
			return 0,0
			
		if y_min2 > y_max1:
			if verbose: print('after y')
			return 0,0
		
		if x_max2 < x_min1:
			if verbose: print('before x')
			return 0,0
		
		if x_min2 > x_max1:
			if verbose:print('after x')
			return 0,0

		y_min = y_min1 if y_min1 >= y_min2 else y_min2
		y_max = y_max1 if y_max1 <= y_max2 else y_max2

		x_min = x_min1 if x_min1 >= x_min2 else x_min2
		x_max = x_max1 if x_max1 <= x_max2 else x_max2

		a0 = self.calc_area(y_min, y_max, x_min, x_max)
		a1 = self.calc_area(y_min1, y_max1, x_min1, x_max1)
		a2 = self.calc_area(y_min2, y_max2, x_min2, x_max2)

		if a1 <= 0:
			a1 = 1
			print("Error a1:", y_min1, y_max1, x_min1, x_max1)

		if a2 <= 0:
			a2 = 1
			print("Error a2:", y_min2, y_max2, x_min2, x_max2)

		perc1 = a0/a1
		perc2 = a0/a2

		if verbose:
			print("min:", y_min, y_max, x_min, x_max)
			print("areas", a0, a1, a2)

		return perc1, perc2

	def select_and_draw_contours(self, img:PILImage.Image, imgray:PILImage.Image, min_contours:int=50, max_contours:int=2000, 
		 						 min_area:int=100, max_area:int=90000, 
		 						 start_colors:tuple=(10, 10, 10), perc_area_threshold:float=.75, 
		 						 font:int=cv2.FONT_HERSHEY_PLAIN, color_text:tuple=(225, 60, 10),
		 						 figsize:tuple=(16,14), cmap:str='coolwarm', 
		 						 ampli_full_text:int=2, ampli_seg_text:int=4, del_text:int=30,
		 						 show_segements:bool=False, show_image:bool=True, verbose:bool=False):

		contours, hierarchy = cv2.findContours(imgray, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

		external_contours = np.zeros(imgray.size)

		red, green, blue = start_colors

		icount=-1; dic_img, dic_img_ori={},{}


		for i in range(len(contours)):

			# last column in the array is -1 if an external contour (no contours inside of it)
			if hierarchy[0][i][3] != -1:
				continue

			# We can now draw the external contours from the list of contours
			n = len(contours[i])

			if n >= min_contours and n <= max_contours:
				continue

			x_min, x_max, y_min, y_max = self.define_xy_min_max(contours[i])
			area = self.calc_area(y_min, y_max, x_min, x_max)
			if area < min_area or area > max_area:
				# print(f"Area(1) {area}:", y_min, y_max, x_min, x_max)
				continue

			# decrementing min, incrementing max --> build a larger area
			x0 = int((x_min+x_max)/2)
			y0 = int((y_min+y_max)/2)

			x_min2 = x0-self.deltax
			if x_min2 < 0:
				x_min2=0
				x0 += self.deltax

			x_max2 = x0+self.deltax
			if x_max2 > self.maxi_x0m1: 
				x_max2 = self.maxi_x0m1
				x_min2 = self.maxi_x0m1 - (2*self.deltax)
				if x_min2 < 0:
					x_min2 = 0

			y_min2 = y0-self.deltay
			if y_min2 < 0:
				y_min2=0
				y0 += self.deltay

			y_max2 = y0+self.deltay
			if y_max2 > self.maxi_y0m1: 
				y_max2 = self.maxi_y0m1
				y_min2 = self.maxi_y0m1 - (2*self.deltay)
				if y_min2 < 0:
					y_min2 = 0

			area = self.calc_area(y_min2, y_max2, x_min2, x_max2)
			if area < min_area:
				print(f"Area(2) {area}:", y_min2, y_max2, x_min2, x_max2)
				continue

			if icount == -1:
				icount = 0
				dic_img[icount]		= [i, y_min2, y_max2, x_min2, x_max2]
				dic_img_ori[icount] = [i, y_min,  y_max,  x_min,  x_max]
			else:
				has_overlap = False

				coord2 = [y_min2, y_max2, x_min2, x_max2]
				
				for i_img, (idummy, y_min1, y_max1, x_min1, x_max1)  in dic_img.items():

					coord1 = [y_min1, y_max1, x_min1, x_max1]
					perc1, perc2 = self.calc_2_overlaps(coord1, coord2)

					if perc1 > perc_area_threshold and perc2 > perc_area_threshold:
						# print(f"*** overlap {i} {perc1:.2f}, {perc2:.2f}")
						has_overlap = True
						break

				if has_overlap:
					# print(">>> has overlap")
					continue

				# print(">>> no overlap")

				icount += 1
				dic_img[icount]		= [i, y_min2, y_max2, x_min2, x_max2]
				dic_img_ori[icount] = [i, y_min,  y_max,  x_min,  x_max]

			red += 25
			if red > 255:
				red = 0
				green += 25

				if green > 255:
					green = 0
					blue += 25

					if blue > 255:
						blue = 0
					
			color = (red, green, blue)

			if verbose and icount == 5:
				print(">>>", icount, n, "color", color, "contours", contours[i][0][0])
				
			if show_segements:
				external_contours = np.zeros(imgray.size)

			cv2.drawContours(external_contours, contours, i, color, -1)

			if show_image and show_segements:
				_, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,8))
				
				title = f"i {i} has {n} contours, image {icount}, X={x_min2}-{x_max2} Y={y_min2}-{y_max2}"
				plt.suptitle(title)

				box = (x_min2, y_min2, x_max2, y_max2) # left, upper, right, lower
				crop_img = img.crop(box)

				x = x_min-del_text
				y = y_min+del_text
				
				if x < del_text:
					x=del_text
				elif x >= self.maxi_x0-del_text:
					x=self.maxi_x0-del_text

				if y < del_text: 
					y = del_text
				elif y >= self.maxi_y0-del_text: 
					y=self.maxi_y0-del_text

				cv2.putText(external_contours, f'{i}',(x,y), font, ampli_seg_text, color_text,3) 

				ax[0].imshow(external_contours, cmap=cmap)
				ax[1].imshow(crop_img, cmap=cmap)


		if show_image and not show_segements:
			for __build_class__, (i, y_min, y_max, x_min, x_max) in dic_img_ori.items():

				x = x_min
				y = y_min
				
				if x < del_text:
					x=del_text
				elif x >= self.maxi_x0-del_text:
					x=self.maxi_x0-del_text

				if y < del_text: 
					y = del_text
				elif y >= self.maxi_y0-del_text: 
					y=self.maxi_y0-del_text

				cv2.putText(external_contours, f'{i}',(x,y), font, ampli_full_text, color_text,3) 

			self.display_img(external_contours, cmap=cmap, figsize=figsize)

		return dic_img, dic_img_ori, contours



	def save_all_images_multiprocess(self, perc_area_threshold:float=0.4, 
									 min_contours:int=400, max_contours:int=5000, 
									 min_area:int=300, max_area:int=600*600, 
									 ampli_full_text:int=2, ampli_seg_text:int=5, del_text:int=50,
									 figsize:tuple=(16,14), start_colors:tuple=(10, 10, 10),
									 color_text:tuple=(225, 60, 10), show_segements:bool=False,
									 Ninf:int=28, Nmid:int=80, Nmid_val:int=60, Nmax:int=127,
									 show_image:bool=False, force:bool=False, verbose:bool=False):

		files = os.listdir(self.root_img)

		for fname_img in files:
			img = self.read_PIL_image(fname_img=fname_img, verbose=True)
			print(f"shape {img.size}")  # if verbose: 

			imgray = self.convert_img_to_gray(img)

			print("reducing grays ...")
			imgray2 = self.reduce_to_3_gray_patterns(imgray, Ninf, Nmid_val, Nmid, Nmax)


			# dic_img, dic_img_ori, contours = \
			dic_img, _, _ = \
			self.select_and_draw_contours(img, imgray2, min_contours=min_contours, max_contours=max_contours,
				 						  min_area=min_area, max_area=max_area,
										  start_colors=start_colors, perc_area_threshold=perc_area_threshold, 
				 						  font=cv2.FONT_HERSHEY_PLAIN, color_text=color_text,
				 						  figsize=figsize, cmap='coolwarm', 
										  ampli_full_text=ampli_full_text, ampli_seg_text=ampli_seg_text, del_text=del_text,
										  show_segements=show_segements, show_image=show_image, verbose=verbose)

			_ = self.save_image_parallel(dic_img=dic_img, img=img, fname_img=fname_img, process_name='save_image', cpus=6, verbose=True)

		print("\n------------------- end (final) ----------------------\n")




	def save_image_parallel(self, dic_img:dict, img:PILImage.Image, fname_img:str,
							process_name='save_image', cpus:int=6, 
							force:bool=False, verbose:bool=False) -> List:

		new_size = (self.deltax*2, self.deltay*2)
		self.new_size = new_size

		par = Parallel(ima=self, dic_img=dic_img, img=img, 
					   fname_img=fname_img, root_save_img=self.root_save_img,
					   process_name=process_name, cpus=cpus, new_size=new_size, verbose=verbose)

		self.par = par

		ret_list = par.run_multiprocess(force=force, verbose=verbose)

		print("\n----------------- end ------------------------\n")
		return ret_list



	def set_filelog(self, fname:str, root:str='./logs'):

		if not os.path.exists(root):
			os.mkdir(root)

		filefull = osjoin(root, fname)
		self.filelog = filefull

		if os.path.exists(filefull):
			os.unlink(filefull)


	def log_save(self, stri:str, withtime:bool=True, 
				 shift:int=0, crBefore:bool=False, 
				 withCR:bool=True, end:str='\n', verbose=False):

		if shift > 0:
			try:
				stri = "\t"*int(shift) + stri
			except:
				pass

		if crBefore:
			stri = "\n" + stri

		if verbose:
			print(stri, end=end)

		if withtime:
			date_time = datetime.now().strftime("%H:%M:%S, %Y/%m/%d\n")
			stri += " >> " + date_time

		if withCR:
			stri += "\n"

		try:
			f = open(self.filelog,"a+")
			f.write(stri)
		except:
			print(f"Could not save on '{self.filelog}'")
		finally:
			f.close()


