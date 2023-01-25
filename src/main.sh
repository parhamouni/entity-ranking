#python data_preprocessing.py
# CUDA_VISIBLE_DEVICES=0 python trainer.py  --max_epochs 1 --enable_model_summary --log_every_n_steps 50 --gpus 1 --track_grad_norm 1 
# CUDA_VISIBLE_DEVICES=0 python trainer.py  --max_epochs 1 --enable_model_summary --log_every_n_steps 50 --gpus 1 --track_grad_norm 1  --biggraph_embedding
CUDA_VISIBLE_DEVICES=1 python trainer.py  --max_epochs 1 --enable_model_summary --log_every_n_steps 50 --gpus 1 --track_grad_norm 1  --biggraph_embedding --deepct
CUDA_VISIBLE_DEVICES=1 python trainer.py  --max_epochs 1 --enable_model_summary --log_every_n_steps 50 --gpus 1 --track_grad_norm 1  --deepct
