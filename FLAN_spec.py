from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
# outputs = model.generate(**inputs)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
print(model)
print("")
print(model.config)
print("")
print(model.encoder.parameters())
print("")
print(model.get_input_embeddings())