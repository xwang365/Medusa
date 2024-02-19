model='FasterDecoding/medusa-vicuna-7b-v1.3' 
base_model="lmsys/vicuna-7b-v1.3"
model_path="/mnt/wx/.cache/huggingface/hub/models--FasterDecoding--medusa-vicuna-7b-v1.3/snapshots/82ac200bf7502419cb49a9e0adcbebe3d1d293f1/"
source ../hf_mirror.sh
#huggingface-cli download --resume-download  --token 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 $model
#huggingface-cli download --resume-download  --token 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 $base_model
export CUDA_VISIBLE_DEVICES=7
# python -m medusa.inference.cli --model $model_path
python -m medusa.inference.inference_test --model $model_path