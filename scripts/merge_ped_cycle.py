#!/usr/bin/env python
from pathlib import Path
import os
import sys
import numpy as np
from tqdm import tqdm
import ipdb as pdb

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
				if line[0] == obj and float(line[15]) > thres:
					line_str = ' '.join(line[:16]) + '\n'
					out_f.write(line_str)
			out_f.close()



if __name__ == '__main__':

	if len(sys.argv)<2:
		raise Exception('Usage: python merge_results.py results_dir')
	class_names = ['Pedestrian']
	results_dir = sys.argv[1]
	thres = 0
	merge_labels(Path(results_dir), class_names, thres)
