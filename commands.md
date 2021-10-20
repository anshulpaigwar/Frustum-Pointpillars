export PYTHONPATH="${PYTHONPATH}:/home/anshul/es3cap/my_codes/frustum_pp/second.pytorch"
python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=ckpt/frustum_pp_car/

To generate final results in kitti format:
python pytorch/train.py evaluate --config_path=configs/pointpillars/car/xyres_16.proto --model_dir=ckpt/first_split/ --ckpt_path=ckpt/first_split/voxelnet-324800.tckpt --pickle_result=False --predict_test=True --det_dir=/home/anshul/es3cap/kitti_data/testing/label_car/

To offline evalute the results:
 ./evaluate_object_3d_offline /home/anshul/es3cap/kitti_data/testing/clean_pvrcnn/ /home/anshul/es3cap/kitti_data/testing/fppg_car_results/unmesh_results/

Get concise values of the results:
python parser.py /home/anshul/es3cap/kitti_data/testing/fppg_car_results/unmesh_results/
