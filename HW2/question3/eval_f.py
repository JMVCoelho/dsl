import evaluate # Huggingface evaluatetokenizer
import numpy as np

import warnings
warnings.filterwarnings("ignore")

bleu_metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels, input_ids):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    input_ids = [[input_id.strip()] for input_id in input_ids]

    return preds, labels, input_ids


def compute_metrics(output_dir, tokenizer, eval_preds):
    preds, labels = eval_preds # Check the location of input_ids is appropriate - 
    #I removed input_ids because i'll predict with the "generate" hf interface.
    
    # Preds
    if isinstance(preds, tuple):
        preds = preds

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   
    # Store inference
    with open(output_dir+'/transliterations.txt','w', encoding='utf8') as wf:
         for translation in decoded_preds:
            wf.write(translation.strip()+'\n') 

    #Labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #print("decoded_labels: ", decoded_labels[:5])
    error_rates = []
    for prediction, label in zip(decoded_preds, decoded_labels):
        true_len = len(label)
        error_rate = distance(label, prediction) / true_len
        error_rates.append(error_rate)
            
        
    bleu = bleu_metric.compute(references=decoded_labels, predictions=decoded_preds, tokenize = 'char')
        
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result = {}
    result["gen_len"] = np.mean(prediction_lens)
    result["error_rate"] = np.mean(error_rates)
    result["bleu"] = round(bleu["score"], 2)
    result = {k: round(v, 4) for k, v in result.items()}
    print(result)

    # Store the score
    with open(output_dir+'/test_score.txt','w', encoding='utf8') as wf:
        for key, value in result.items():
            wf.write(f"{key}: {value}\n") #ensure_ascii=False

    return result


def distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    m = np.zeros([len(str2) + 1, len(str1) + 1], dtype=int)
    for x in range(1, len(str2) + 1):
        m[x, 0] = m[x - 1, 0] + 1
    for y in range(1, len(str1) + 1):
        m[0, y] = m[0, y - 1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y - 1] == str2[x - 1]:
                dg = 0
            else:
                dg = 1
            m[x, y] = min(
                m[x - 1, y] + 1, m[x, y - 1] + 1, m[x - 1, y - 1] + dg
            )
    return m[len(str2), len(str1)]



