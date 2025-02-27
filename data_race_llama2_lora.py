# -------------------- Cell --------------------
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
import ast

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
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
# tokenizer = RobertaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# model = RobertaForSequenceClassification.from_pretrained(
#     "meta-llama/Llama-2-7b-hf", num_labels=2
# )

# -------------------- Cell --------------------
# Assuming 'data' is in Dataset format
from datasets import Dataset

# Convert data to Dataset format if needed
dataset = Dataset.from_dict({
    'prompt': [item['prompt'] for item in data],
    'response': [item['response'] for item in data],
})
print(dataset[0])
# -------------------- Cell --------------------
device_map="FSDP"
# -------------------- Cell --------------------
# Split into train and test datasets
train_test_split = dataset.train_test_split(test_size=0.2)  # Adjust test_size as needed

# Accessing train and test datasets
train_ds = train_test_split['train']
test_ds = train_test_split['test']

# -------------------- Cell --------------------
def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['prompt']}\n\nAnswer: {example['response']}"
    return text

# -------------------- Cell --------------------
from trl.trainer import ConstantLengthDataset
import torch
# Use the preformatted dataset with tokenized inputs
train_dataset = ConstantLengthDataset(
    tokenizer,
    train_ds,
    formatting_func=prepare_sample_text,
    infinite=True,
    seq_length=1024
)
test_dataset = ConstantLengthDataset(
    tokenizer,
    test_ds,
    formatting_func=prepare_sample_text,
    infinite=True,
    seq_length=1024
)



# Show one sample from train set
iterator = iter(train_dataset)
sample = next(iterator)
print(sample)

# -------------------- Cell --------------------
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# -------------------- Cell --------------------
from transformers import TrainingArguments
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
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf',torch_dtype=torch.bfloat16)
# -------------------- Cell --------------------
import wandb
wandb.init(
    project="data_race_detection",
    entity="redxnessrine-redx",
    name="data_race_llama2_lora",
    resume="allow",
    mode="online"
)
# -------------------- Cell --------------------
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=lora_config,
    packing=True,
)
# -------------------- Cell --------------------
# handle PEFT+FSDP case
trainer.model.print_trainable_parameters()
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
# -------------------- Cell --------------------
print("Training...")
trainer.train()
print("Training complete")
# -------------------- Cell --------------------
