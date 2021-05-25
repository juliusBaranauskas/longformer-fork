
from transformers import LongformerTokenizer, LongformerForQuestionAnswering
import torch
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
# question, text = "Did Anne like Jones?", "Even though Anne did not like Jones, she insisted on staying. After all, Jim Henson was quite tolerable for a simp"

f = open("text.txt", "r")

lines = f.readlines()
f.close()
questions = []
contexts = []

for line in lines:
    tupl = line.split(",")
    contexts.append(tupl[0])
    questions.append(tupl[1])
    

for i in range(len(contexts)):
    
    print(contexts[i])

    encoding = tokenizer(questions[i], contexts[i], return_tensors="pt")
    input_ids = encoding["input_ids"]
    # default is local attention everywhere
    # the forward method will automatically set global attention on question tokens
    attention_mask = encoding["attention_mask"]
    outputs = model(input_ids, attention_mask=attention_mask)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer_tokens = all_tokens[torch.argmax(start_logits) :torch.argmax(end_logits)+1]
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens)) # remove space prepending space token
    print("Answer: " + answer)