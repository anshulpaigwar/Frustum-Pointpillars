#!/usr/bin/env python
from pathlib import Path
import os
import sys
import numpy as np
from tqdm import tqdm
import ipdb as pdb

def cleanup_labels(result_dir, out_dir, thres):
	list = os.listdir(result_dir) # dir is your directory path
	for idx in tqdm(range(len(list))):
		
		label_file = result_dir + '/' + '%06d.txt' % idx
		with open(label_file, 'r') as f:
			lines = f.readlines()
		
		out_file = out_dir + '/' + '%06d.txt' % idx
		out_f = open(out_file, 'w')  # open file in append mode
		content = [line.strip().split(' ') for line in lines]
		for line in content:
			if float(line[15]) > thres:
				line_str = ' '.join(line[:15]) + '\n'
				out_f.write(line_str)
		out_f.close()



if __name__ == '__main__':

	if len(sys.argv)<3:
		raise Exception('Usage: python cleanup_results.py results_dir out_dir')

	result_dir = sys.argv[1]
	out_dir = sys.argv[2]
	cleanup_labels(result_dir, out_dir, thres = 0.5)
