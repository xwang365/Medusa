export CUDA_VISIBLE_DEVICES=7
export port=12306
export model_dir="/mnt/wx/.cache/huggingface/hub/models--FasterDecoding--medusa-vicuna-7b-v1.3/snapshots/82ac200bf7502419cb49a9e0adcbebe3d1d293f1/"
python3 -m medusa.inference.api_server --host 0.0.0.0 --port $port --model $model_dir | tee  server.log

