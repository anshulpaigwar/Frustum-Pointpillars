#!/usr/bin/env python
from pathlib import Path
import os
import sys
import numpy as np
from tqdm import tqdm
import ipdb as pdb


def yolo_to_kitti_labels(yolo_results_dir, out_dir, thres):
	list = os.listdir(yolo_results_dir)  # dir is your directory path
	yolo_classes = ["Car", "Person", "Bicycle"]
	kitti_classes = ["Car", "Pedestrian", "Cyclist"]

	for idx in tqdm(range(7518)):
		lines = []
		label_file = yolo_results_dir + '/' + '%06d.txt' % idx
		out_file = out_dir + '/' + '%06d.txt' % idx
		if os.path.isfile(label_file):
			with open(label_file, 'r') as f:
				lines = f.readlines()

			out_f = open(out_file, 'w')  # open file in append mode
			content = [line.strip().split(' ') for line in lines]
			for line in content:
				obj_class = line[0]
				xmin = float(line[-5])
				ymin = float(line[-4])
				xmax = float(line[-3])
				ymax = float(line[-2])
				score = float(line[-1])
				if obj_class in yolo_classes and score > thres:
					i = yolo_classes.index(obj_class)
					out_f.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n" % (kitti_classes[i], xmin, ymin, xmax, ymax, score))
			out_f.close()
		else:
			open(out_file, 'a').close()
			print('%06d.txt' % idx)


    # ref_dets = read_det_file(ref_det_file)
    # id_list = ref_dets.keys()
    # for image_idx in id_list:
    #     label_file = results_dir / ('%06d.txt' % (image_idx))
    #     with open(str(label_file), 'w') as f:
    #         frame_det_dict = ref_dets.get(image_idx, None)
    #         if frame_det_dict:
    #             ref_names = frame_det_dict["names"]
    #             ref_bboxes = frame_det_dict["bboxes"]
    #             ref_scores = frame_det_dict["prob"]
    #             # pdb.set_trace()

    #             for label, bbox, score in zip(ref_names, ref_bboxes, ref_scores):

    #                 xmin = bbox[0]
    #                 ymin = bbox[1]
    #                 xmax = bbox[2]
    #                 ymax = bbox[3]
    #                 f.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n" % (label, xmin, ymin, xmax, ymax, score))
    #         else:
    #             print('no detection for image idx:', image_idx)










# def cleanup_labels(result_dir, out_dir, thres):
# 	list = os.listdir(result_dir) # dir is your directory path
# 	for idx in tqdm(range(len(list))):
		
# 		label_file = result_dir + '/' + '%06d.txt' % idx
# 		with open(label_file, 'r') as f:
# 			lines = f.readlines()
		
# 		out_file = out_dir + '/' + '%06d.txt' % idx
# 		out_f = open(out_file, 'w')  # open file in append mode
# 		content = [line.strip().split(' ') for line in lines]
# 		for line in content:
# 			if float(line[15]) > thres:
# 				line_str = ' '.join(line[:15]) + '\n'
# 				out_f.write(line_str)
# 		out_f.close()



if __name__ == '__main__':

	# if len(sys.argv)<3:
	# 	raise Exception('Usage: python cleanup_results.py results_dir out_dir')

	# result_dir = sys.argv[1]
	# out_dir = sys.argv[2]
	yolo_results_dir = "/home/anshul/es3cap/kitti_data/testing/labels_yolo_v5/labels/"
	out_dir = "/home/anshul/es3cap/kitti_data/testing/labels_yolo_v5/labels_yolo_kitti_format/"
	yolo_to_kitti_labels(yolo_results_dir, out_dir, thres = 0.5)
