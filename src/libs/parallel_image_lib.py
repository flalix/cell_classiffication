#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
# Created on 2025/01/26
# Udated  on 2025/01/26
# @author: Flavio Lichtenstein
# @local: Bioinformatics: CENTD/Molecular Biology; Instituto Butatan


import multiprocessing as mp
# import multiprocess as mp
import cv2

import sys, os, random # , copy
import numpy as np
from datetime import datetime
from typing import Optional, Iterable, Set, Tuple, Any, List

# import PIL
from PIL import Image as PILImage

# import matplotlib.pyplot as plt

class Parallel:

	def __init__(self, ima, dic_img:dict, img:PILImage.Image, fname_img:str, root_save_img:str, 
				 process_name:str='save_image', cpus:int=6, new_size:tuple=(500,500), verbose:bool=True):

		self.ima = ima
		self.dic_img = dic_img
		self.img = img
		self.process_name = process_name

		self.fname_img = fname_img
		self.root_save_img = root_save_img

		self.new_size = new_size

		mat = self.fname_img.split('.')[:-1]
		fname_img_seg	  = "_".join(mat).replace(' ','_')
		self.fname_img_seg = fname_img_seg + '_segment_%d.png'

		maxcpu = mp.cpu_count()
		if cpus is None or cpus == 0 or cpus > maxcpu:
			cpus = maxcpu-2
		else:
			if cpus > maxcpu-2:
				cpus = maxcpu-2

		self.cpus = cpus

		stri = f">>> Parallel {process_name} cpus={cpus}"
		self.ima.log_save(stri, verbose=verbose)

		self.worker = self._parallel_save_worker



	def _parallel_save_worker(self, pid:int, key_list:List, force:bool=False, verbose:bool=False) -> bool:

		for key in key_list:

			i, y_min2, y_max2, x_min2, x_max2 = self.dic_img[key]

			fnamefig = self.fname_img_seg%(i)
			fullname = os.path.join(self.root_save_img, fnamefig)

			if os.path.exists(fullname) and not force:
				print(f"ok {pid} {key} {i} ....", end=' ')
				return True

			try:
				box = (x_min2, y_min2, x_max2, y_max2) # left, upper, right, lower
				crop_img = self.img.crop(box)	
				# print(">>>", crop_img.shape, y_min2, y_max2, x_min2, x_max2)
				print(f"saving {pid} {key} {i} shape {crop_img.size}....", end=' ')
				crop_img_resized = cv2.resize(crop_img, self.new_size, interpolation=cv2.INTER_AREA)  # Good for shrinking
				# print("writing ....", end=' ')
				cv2.imwrite(fullname, crop_img_resized)
				#print(f"saved {pid} {key} {i}", end=' ')
			except:
				print(f"Error: file not saved {pid} {key} {i}: {fullname}")
				# raise Exception("stop")
				return False

		return True


	def run_multiprocess(self, force:bool=False, verbose:bool=False):
		'''
		Call function with multiprocessing pools. Used for saving image
		in parallel where the main input is dic_img
		Args:
			dic_img
			cpus: is in the constructor
			worker: save_image function
		Returns:
			True or False: saved or not
		   '''
		pool = mp.Pool(self.cpus)
		funclist = []

		key_list = list(self.dic_img.keys())
		random.shuffle(key_list)

		chunks = np.array_split(np.array(key_list, dtype=np.ndarray), self.cpus)

		for pid, chunk in enumerate(chunks):
			print(">>> pid:", pid)

			try:
				f = pool.apply_async(self.worker, np.array([pid, chunk, force, verbose], dtype=np.ndarray))
				funclist.append(f)

			except KeyboardInterrupt:
				stri = 'Error: process was interrupted by keyboard'
				self.ima.log_save(stri, verbose=True)
				return [False]
			except Exception as e:
				stri = 'Error: process has problem: %s'%(str(e))
				self.ima.log_save(stri, verbose=True)
				return [False]

		result_list = []

		for f in funclist:
			ret = f.get(timeout=None)
			result_list.append(ret)

		pool.close()
		pool.join()

		return result_list

