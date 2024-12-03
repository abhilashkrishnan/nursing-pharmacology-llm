max_seq_length = 2048
dtype = None
load_in_4bit = True

nursing_prompt = """Below is an question with a set of operations to be performed by nurses, paired with set of operations and an explaination that provides further context. Write a response that appropriately complete the request.

### Question
{}

### Operation a
{}

### Operation b
{}

### Operation c
{}

### Operation d
{}

### Operation e
{}

### Operation f
{}

### Explanation
{}

"""

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
        "", # op_d,
        "", # op_e,
        "", # op_f,
        "", # Explanation - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
