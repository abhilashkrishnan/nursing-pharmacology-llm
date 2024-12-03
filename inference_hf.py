from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("abhilashkrish/nursing-pharmacology")
model = AutoModelForCausalLM.from_pretrained("abhilashkrish/nursing-pharmacology")

# Generate text
input_text = "The nurse is preparing to administer a schedule II injectable drug and is drawing up half of the contents of a single-use vial. Which nursing action is correct?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
