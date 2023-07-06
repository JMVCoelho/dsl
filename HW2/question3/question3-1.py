###
### DSL 2023 HW2.3 implemented code
###
 

import argparse
from transformers import  Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, DataCollatorForSeq2Seq
import numpy as np
from functools import partial
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
from datasets import disable_caching
disable_caching()

import eval_f
import string
from charactertokenizer import CharacterTokenizer
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, PretrainedConfig

from datasets import load_dataset, concatenate_datasets # Huggingface datasets


from datasets import disable_caching
disable_caching()


# helper function for reading arguments with a parser
# convenient for use with a yaml config file
def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for training models.")
    parser.add_class_arguments(Seq2SeqTrainingArguments, "training_args")
    parser.add_class_arguments(EarlyStoppingCallback, "early_stopping")
    parser.add_argument("--cfg",action=ActionConfigFile) 
    return parser 



def preprocess_function(tokenizer, data): 
    """
    tokenizer: initialised tokenizer instance
    data: loaded dataset
    """
    # Question 3.1

    inputs = tokenizer(data['text'], padding=True, truncation='longest_first', return_tensors='pt')
    labels = tokenizer(text_target=data['labels'], padding=True, truncation='longest_first', return_tensors='pt')

    inputs['labels'] = labels['input_ids']
    return inputs
    

def compute_vocabulary(data):
    """implement this function to return the vocabulary for your tokenizer and model

    Args:
        data (dict): loaded dataset
    """
    # Question 3.1
    # 84 individual tokens (+7 special ones)

    tokens = []
    splits = ["train"]

    for split in splits:
        for example in data[split]:
            ar = example['labels']
            en = example['text']

            example_tokens = [*ar] + [*en]

            for token in example_tokens:
                if token not in tokens:
                    tokens.append(token)

    print(len(tokens))
    return tokens



class CustomConfig(BertConfig):
    
    def __init__(self,model_type:str='bert-base-uncased', **kwargs,):
        super().__init__(**kwargs)
    
        """This class needs to be expanded to adapt the hyperparameters based on the exercise
        """

        self.vocab_size = kwargs['vocab_size']
        self.hidden_size = 384
        self.num_hidden_layers = 4
        self.num_attention_heads = 6
        self.intermediate_size = 1024
        self.max_position_embeddings = 128

        # Question 3.1
    
    
def main():
    parser = read_arguments()
        
    cfg = parser.parse_args(["--cfg", "configuration.yaml"])
    output_dir = cfg.training_args.output_dir    

    data_files = {"train": "../../data/ar2en-train.txt", "val": "../../data/ar2en-eval.txt", "test": "../../data/ar2en-test.txt"}

    dataset = load_dataset('csv', delimiter="\t", data_files=data_files, column_names=['text', 'labels']) #path to the folder where you downloaded the data

    vocab = compute_vocabulary(dataset)
    
    model_max_length = 20
    tokenizer = CharacterTokenizer(vocab, model_max_length)

    config_encoder = CustomConfig(vocab_size=tokenizer.vocab_size)
    config_decoder = CustomConfig(vocab_size=tokenizer.vocab_size, decoder_start_token_id=tokenizer.cls_token_id, force_bos_token_to_be_generated=True)

    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    #print(config)
   
    model = EncoderDecoderModel(config=config)
    
    tokenized_datasets = dataset.map(
        partial(preprocess_function, tokenizer),
        batched=True,
        remove_columns="text", 
        )

    # Create a batch using DataCollator and pad dinamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt") 

    model.config.decoder_start_token_id = tokenizer._convert_token_to_id('[CLS]')
    model.config.pad_token_id = tokenizer._convert_token_to_id('[PAD]')

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        eval_steps=1,
        evaluation_strategy = "epoch",
        save_strategy= "epoch",
        per_device_train_batch_size=cfg.training_args.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training_args.per_device_train_batch_size,
        learning_rate=cfg.training_args.learning_rate,
        num_train_epochs=cfg.training_args.num_train_epochs,
        load_best_model_at_end = True,
        metric_for_best_model = cfg.training_args.metric_for_best_model,
        greater_is_better = cfg.training_args.greater_is_better,
        seed=1,
        disable_tqdm=False,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,                         
        args=training_args, #need to be implemented to add the training arguments              
        train_dataset=tokenized_datasets["train"],     
        eval_dataset=tokenized_datasets["val"],        
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(eval_f.compute_metrics, output_dir, tokenizer),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping.early_stopping_patience)],
        )

    """Implement code to train and evaluate the model using the trainer
    """
    trainer.train()

    trainer.save_model(output_dir)
    trainer.save_state()

    predictions = trainer.predict(tokenized_datasets["test"])

    print(predictions)



if __name__ == "__main__":
    main()
