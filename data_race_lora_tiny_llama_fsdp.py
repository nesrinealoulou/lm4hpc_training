import json
import wandb
from trl import SFTTrainer
import torch
import ast
from peft import LoraConfig
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft.utils.other import fsdp_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# Load JSON data and ensure all prompts are strings
with open('../data_preprocessing/simple_prompt.json', 'r') as file:
    data = json.load(file)
print(data[:1])
# -------------------- Cell --------------------
def convert_prompt_to_format(data):
    result = []
    for item in data:
        # The first part of the prompt (the intro message)
        prompt = item['prompt']
        response = item['response']
        
        # Find the index where the phrase "if absent" ends
        separator = "if absent."
        split_point = prompt.find(separator)
        
        if split_point != -1:
            # Get the first part (before "if absent") and strip
            intro = prompt[:split_point + len(separator)].strip()
            # Get the second part (after "if absent") which should be the code
            code = prompt[split_point + len(separator):].strip()
        else:
            # If "if absent" is not found, just return the prompt as is
            intro = prompt
            code = ""
        
        # Format the result as a dictionary
        formatted_data = {
            joined_code = ''.join(code_list).replace('\\n', '\n').strip()  # Fix escape characters
            'prompt': intro,  # The intro part of the prompt (before the code)
            'code': code,
             'response': response
        }
        if formatted_data['code'] == '': 
            print('empty')
        
        result.append(formatted_data)
    
    return result

# Applying the function to convert the data
formatted_data = convert_prompt_to_format(data)

# Printing the result to see the converted format
for item in formatted_data[:5]:
    print(item)

# -------------------- Cell --------------------
# Function to process and combine the prompt string with the code list
def preprocess_prompt(example):
    try:
        # Safely evaluate the 'code' field to convert it to a list
        code_list = ast.literal_eval(example['code']) 
    except (ValueError, SyntaxError) as e:
        # Handle malformed 'code' list, e.g., invalid escape characters or other issues
        print(f"Error evaluating code field for example: {example['prompt']}")
        code_list = []

    # If the code list is valid, join it and append to the prompt
    if isinstance(example['prompt'], str) and isinstance(code_list, list):
        if code_list:
            
            # Ensure the code is joined properly and handle escape characters
            joined_code = ''.join(code_list).replace('\\n', '\n').strip()  # Fix escape characters
            example['prompt'] = example['prompt'] + "\n" + joined_code
        
        # Remove the 'code' field after transformation
        del example['code']

    return example

# Process each example in the data
for item in formatted_data:
    preprocess_prompt(item)
    #print(f"Processed prompt: {item['prompt'][:500]}")  # Print first 500 characters to verify the format
data = formatted_data 
print('data', data[:5])

# -------------------- Cell --------------------
# Convert data to Dataset format if needed
dataset = Dataset.from_dict({
    'prompt': [item['prompt'] for item in data],
    'response': [item['response'] for item in data],
})
print(dataset[0])
# -------------------- Cell --------------------
device_map="FSDP"
# -------------------- Cell --------------------
tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
trust_remote_code=True,
cache_dir='',
use_cache = False,
torch_dtype = getattr(torch, "bfloat16"),)


# -------------------- Cell --------------------

def preprocess_dataset(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512)
# Apply tokenization
dataset = dataset.map(preprocess_dataset, batched=True)
# Remove original text fields (optional but helps avoid issues)
dataset = dataset.remove_columns(["prompt", "response"])



# -------------------- Cell --------------------
lora_alpha = 8
lora_dropout = 0.1
lora_r = 32
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projection layers
        "gate_proj", "up_proj", "down_proj",  # MLP layers
        "input_layernorm", "post_attention_layernorm"  # Normalization layers
    ],
    modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
)
# -------------------- Cell --------------------
max_seq_length = 512
output_dir = "./results"
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
optim = "adamw_torch"
save_steps = 10
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    bf16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = {"use_reentrant": True},
    report_to="wandb",
)
# -------------------- Cell --------------------
wandb.init(
    project="data_race_detection",
    entity="redxnessrine-redx",
    name="data_race_llama2_lora",
    resume="allow",
    mode="online"
    )
# -------------------- Cell --------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,

    )
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# -------------------- Cell --------------------
trainer.model.print_trainable_parameters()
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

# -------------------- Cell --------------------
print("Training...")
trainer.train()
print("Training complete")
