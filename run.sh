export PYTHONWARNINGS="ignore"
export GPU=0

python coldl.py --config config/aircraft.cfg --gpu ${GPU} --net1 resnet50 --net2 resnet50 --batch_size 16 --lr 0.003
