from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer, GenerationConfig
import torch
import evaluate

# Global stuff (should be sys.args)

#model_name = "t5_base"
model_name = "t5-small"

inference_batch_size = 64 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
metric_bleu = evaluate.load("sacrebleu")

model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()

# Define task and get task specific parameters 
task = "translation_en_to_fr"

task_prefix = model.config.task_specific_params[task]['prefix']
task_early_stopping = model.config.task_specific_params[task]['early_stopping']
task_num_beams = model.config.task_specific_params[task]['num_beams']
task_max_new_tokens = model.config.task_specific_params[task]['max_length']
task_model_max_len = model.config.task_specific_params[task]['max_length']

# Load dataset and tokenizer
dataset = load_dataset("iwslt2017", 'iwslt2017-en-fr')
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=task_model_max_len)

generation_config = GenerationConfig(max_new_tokens=task_max_new_tokens, num_beams=task_num_beams, early_stopping=task_early_stopping)

# Get text and add prefix
test_set = dataset['test']

texts = [task_prefix + text['en'] for text in test_set['translation']]
labels = [text['fr'] for text in test_set['translation']]


# auxiliary function
def batch(X, batch_size=1):
    l = len(X)
    for idx in range(0, l, batch_size):
        yield X[idx:min(idx + batch_size, l)]

# Generate
outputs = []
for sample in batch(texts, inference_batch_size):
    inputs = tokenizer(sample, padding=True, truncation='longest_first', return_tensors='pt')

    for name in inputs:
        inputs[name] = inputs[name].to(device)

    sample_outputs = model.generate(**inputs, generation_config=generation_config)
    outputs.extend(sample_outputs)

text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text_output[:10])
bleu = metric_bleu.compute(predictions=text_output, references=labels)
print(bleu)

