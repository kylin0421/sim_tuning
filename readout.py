import torch

ckpt_path1='/srv/data/sul/sim_tuning/P-tuning/out/LAMA/prompt_model/megatron_11b/search/megatron_11b_shared_only_evaluate/P36/epoch_0_dev_12.0_test_8.2803.ckpt'
ckpt_path2='/srv/data/sul/sim_tuning/P-tuning/out/LAMA/prompt_model/megatron_11b/search/megatron_11b_shared_only_evaluate/P36/epoch_0_dev_26.6667_test_20.1699.ckpt'


ckpt1=torch.load(ckpt_path1, weights_only=False)  
ckpt2=torch.load(ckpt_path2, weights_only=False)  
print(ckpt1.keys())    #dict_keys(['embedding', 'dev_hit@1', 'test_hit@1', 'test_size', 'ckpt_name', 'time', 'args'])

print(ckpt1['args'])
print(ckpt1['test_hit@1'])
print(ckpt2['args'])
print(ckpt2['test_hit@1'])

