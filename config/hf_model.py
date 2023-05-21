from transformers import TrainingArguments



model_name = 'aubmindlab/bert-base-arabertv02-twitter'

max_len = 90

label_map = {'LY': 0, 'MA': 1, 'EG': 2, 'LB': 3, 'SD': 4}



training_args = TrainingArguments( 
    output_dir= "../models/train_hf",    
    adam_epsilon = 1e-8,
    learning_rate = 2e-5,
    fp16 = True, # enable this when using V100 or T4 GPU
    per_device_train_batch_size = 16, # up to 64 on 16GB with max len of 128
    per_device_eval_batch_size = 128,
    gradient_accumulation_steps = 2, # use this to scale batch size without needing more memory
    num_train_epochs= 10,
    warmup_ratio = 0,
    do_eval = True,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True, # this allows to automatically get the best model at the end based on whatever metric we want
    metric_for_best_model = 'macro_f1',
    greater_is_better = True,
    seed = 25,
    # weight_decay=1e-2

  )