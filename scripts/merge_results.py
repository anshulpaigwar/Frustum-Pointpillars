#!/usr/bin/env python
from pathlib import Path
import os
import sys
import numpy as np
from tqdm import tqdm
import ipdb as pdb

# def merge_labels(result_dir):
# 	car_dir = result_dir / "car"
# 	ped_dir = result_dir / "ped_cycle"
# 	# cyclist_dir = result_dir / "cyclist"
# 	out_dir = result_dir / "merged"
# 	list = os.listdir(str(car_dir)) 
# 	for idx in tqdm(range(len(list))):
		
# 		car_file = car_dir / ('%06d.txt' % idx)
# 		ped_file = ped_dir / ('%06d.txt' % idx)
# 		# cyclist_file = cyclist_dir / ('%06d.txt' % idx)
# 		with open(car_file, 'r') as f:
# 			car_lines = f.readlines()

# 		with open(ped_file, 'r') as f:
# 			ped_lines = f.readlines()

# 		# with open(cyclist_file, 'r') as f:
# 		# 	cyclist_lines = f.readlines()

# 		out_file = out_dir + '/' + '%06d.txt' % idx
# 		out_f = open(out_file, 'w')  # open file in append mode
		
# 		content = [line.strip().split(' ') for line in lines]
# 		for line in content:
# 			if float(line[15]) > thres:
# 				line_str = ' '.join(line[:15]) + '\n'
# 				out_f.write(line_str)
# 		out_f.close()


def merge_labels(results_dir, class_names, thres):
	out_dir = results_dir / "merged"
	for obj in class_names:
		obj_dir = results_dir / obj
		list = os.listdir(str(obj_dir)) 
		for idx in tqdm(range(len(list))):

			obj_file = obj_dir / ('%06d.txt' % idx)
			with open(str(obj_file), 'r') as f:
				lines = f.readlines()

			out_file = out_dir / ('%06d.txt' % idx)
			out_f = open(str(out_file), 'a')  # open file in append mode
		
			content = [line.strip().split(' ') for line in lines]
			for line in content:
				# if line[0] == obj and float(line[15]) > thres:
				line_str = ' '.join(line[:16]) + '\n'
				out_f.write(line_str)
			out_f.close()





if __name__ == '__main__':

	if len(sys.argv)<2:
		raise Exception('Usage: python merge_results.py results_dir')
	class_names = ['car', 'ped_cycle']
	results_dir = sys.argv[1]
	thres = 0
	merge_labels(Path(results_dir), class_names, thres)
