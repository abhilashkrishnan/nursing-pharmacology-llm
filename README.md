# Nursing Pharmacology Large Language Model (LLM)

#### A domain-specific large language model fine-tuned to provide expert-level responses for healthcare training and related applications. Designed to assist healthcare professionals, this model excels at generating precise explanations and adhering to controlled-substance protocols. With over 41 million parameters, it leverages state-of-the-art fine-tuning techniques to optimize for domain specificity and accuracy.

### Model Details

#### Model Description
- Developed by: Abhilash Krishnan
- Model type: Fine-tuned version of meta-llama-3.1-8b-bnb-4bit
- Language(s): English
- License: MIT
- Finetuned from model: unsloth/meta-llama-3.1-8b-bnb-4bit

#### How to Get Started with the Model

##### Here’s a simple example to load and use the model from HuggingFace:

```
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("abhilashkrish/nursing-pharmacology")
model = AutoModelForCausalLM.from_pretrained("abhilashkrish/nursing-pharmacology")

# Generate a response
input_text = "What are the protocols for handling controlled substances?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Model Training Details
- training_details:
  - training_data:
    - dataset: timzhou99/nursing-pharmacology
  - description:
    - The dataset includes text focused on nursing pharmacology and healthcare-specific training materials.
    - training_procedure:
      - preprocessing:
        - Tokenized using Hugging Face's AutoTokenizer with healthcare domain-specific vocabulary.
      - training_regime:
        - Mixed precision with bfloat16 to optimize GPU memory usage and accelerate fine-tuning.
      - hyperparameters:
        - learning_rate: 4e-5
        - batch_size: 8
        - gradient_accumulation_steps: 4
        - epochs: 5
    - compute_infrastructure:
      - hardware: NVIDIA A40 GPU (44 GB VRAM)
      - software:
        - PEFT_library: v0.13.2
        - PyTorch: 2.5.1+cu12
        - CUDA: 12.4
  - evaluation:
    - testing_data:
    - dataset: timzhou99/nursing-pharmacology
  - description:
    - The model was evaluated on a subset of the timzhou99/nursing-pharmacology dataset hosted on HuggingFace using specific healthcare-related tasks.
  - metrics:
    - perplexity:
      - Evaluated as a measure of model fluency.
    - domain_specific_accuracy:
      - Assessed based on its ability to generate accurate healthcare protocols.
```
