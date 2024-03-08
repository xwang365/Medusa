#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=2
export NCCL_P2P_DISABLE=1
#python -m medusa.inference.cli --model /mnt/zhiweit/download/models--FasterDecoding--medusa-vicuna-7b-v1.3/snapshots/82ac200bf7502419cb49a9e0adcbebe3d1d293f1/
python -m medusa.inference.inference_test --model /mnt/zhiweit/download/models--FasterDecoding--medusa-vicuna-7b-v1.3/snapshots/82ac200bf7502419cb49a9e0adcbebe3d1d293f1/
