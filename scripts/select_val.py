#!/usr/bin/env python
from pathlib import Path
import os
import sys
import numpy as np
from tqdm import tqdm
import ipdb as pdb
from shutil import copyfile

# def read_det_file(det_filename):
#     ''' Parse lines in 2D detection output files '''
#     det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
#     id_list = []
#     type_list = []
#     prob_list = []
#     box2d_list = []
#     for line in open(det_filename, 'r'):
#         t = line.rstrip().split(" ")
#         id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
#         type_list.append(det_id2str[int(t[1])])
#         prob_list.append(float(t[2]))
#         box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))

#     all_boxes_2d = {}

#     for i, det_idx in enumerate(id_list):
#         if det_idx not in all_boxes_2d:
#             all_boxes_2d[det_idx] = {'names': [], 'bboxes':[], 'prob':[]}

#         all_boxes_2d[det_idx]['names'].append(type_list[i])
#         all_boxes_2d[det_idx]['bboxes'].append(box2d_list[i])
#         all_boxes_2d[det_idx]['prob'].append(prob_list[i])

#     return all_boxes_2d










def write_labels(data_dir, results_dir, set_file):

    for line in open(set_file, 'r'):
        # pdb.set_trace()
        line = line.rstrip('\n')
        src = data_dir / (line +'.txt')
        dst = results_dir / (line + '.txt')
        copyfile(str(src), str(dst))
        


    # ref_dets = read_det_file(ref_det_file)
    # id_list = ref_dets.keys()
    # for image_idx in id_list:
    #     label_file = results_dir / ('%06d.txt'%(image_idx))
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
    #                 f.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"%(label,xmin,ymin,xmax,ymax,score))
    #         else:
    #             print('no detection for image idx:', image_idx )



# def merge_labels(results_dir, class_names, thres):
#   out_dir = results_dir / "merged"
#   for obj in class_names:
#       obj_dir = results_dir / obj
#       list = os.listdir(str(obj_dir)) 
#       for idx in tqdm(range(len(list))):

#           obj_file = obj_dir / ('%06d.txt' % idx)
#           with open(str(obj_file), 'r') as f:
#               lines = f.readlines()

#           out_file = out_dir / ('%06d.txt' % idx)
#           out_f = open(str(out_file), 'a')  # open file in append mode
        
#           content = [line.strip().split(' ') for line in lines]
#           for line in content:
#               if line[0] == obj and float(line[15]) > thres:
#                   line_str = ' '.join(line[:16]) + '\n'
#                   out_f.write(line_str)
#           out_f.close()



if __name__ == '__main__':

    if len(sys.argv)<3:
        raise Exception('Usage: python select_val.py data_dir results_dir set_file')

    data_dir = sys.argv[1]
    results_dir = sys.argv[2]
    set_file = sys.argv[3]

    write_labels(Path(data_dir), Path(results_dir), set_file)
