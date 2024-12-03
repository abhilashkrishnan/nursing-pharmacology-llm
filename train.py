from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype  = dtype,
    load_in_4bit  = load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None
)

nursing_prompt = """Below is an question with a set of operations to be performed by nurses, paired with set of operations and an explaination that provides further context. Write a response that appropriately complete the request.

### Question
{}

### Operation a
{}

### Operation  b
{}

### Operation c
{}

### Operation d
{}

### Explanation
{}

"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    questions = examples["question"]
    operations_a = examples["op_a"]
    operations_b = examples["op_b"]
    operations_c = examples["op_c"]
    operations_d = examples["op_d"]
    explainations = examples["exp"]
    
    texts = []
    
    for operation_a, operation_b, operation_c, operation_d, operation_e, operation_f, question, explaination in zip(operations_a, operations_b, operations_c, operations_d, operations_e, operations_f, questions, explainations):
        text = nursing_prompt.format(question, operation_a, operation_b, operation_c, operation_d, operation_e, operation_f, explaination) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }
pass

from datasets import load_dataset

dataset = load_dataset("timzhou99/nursing-pharmacology", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none"
    ),
)

trainer_stats = trainer.train()

#Inference
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [
    nursing_prompt.format(
    "The nurse is preparing to administer a schedule II injectable drug and is drawing up half of the contents of a single-use vial. Which nursing action is correct?",
    "Ask another nurse to observe and cosign wasting the remaining drug from the vial.",
    "Keep the remaining amount in the patients drawer to give at the next dose.",
    "Record the amount unused in the patients medication record.",
    "Dispose of the vial with the remaining drug into a locked collection box.",
    "" #Explanation
    )], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

#Save, Loading Fine-tuned model
model.save_pretrained("nurse_lora_model")
tokenizer.save_pretrained("nurse_lora_model")

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "nurse_lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
    nursing_prompt.format(
        "The nurse is preparing to administer a schedule II injectable drug and is drawing up half of the contents of a single-use vial. Which nursing action is correct?", # Question
        "", # op_a
        "", # op_b
        "", # op_c
        "", # op_d
        "" # Explanation - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

