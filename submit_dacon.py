import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from data import DaconVQADataset

import pandas as pd

sample_submission = pd.read_csv('./dacon-vqa/sample_submission.csv')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
model_dir = "./model_checkpoints/best_model"
model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
processor = AutoProcessor.from_pretrained(model_dir)

def collate_fn(batch):
    questions, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    )
    return inputs

# Create datasets
test_dataset = DaconVQADataset("test")

# Create test dataset and DataLoader
test_loader = DataLoader(
    test_dataset, 
    batch_size=32, 
    collate_fn=collate_fn, 
    num_workers=8,
    pin_memory=True,
)

def inference_model(test_loader, model, processor):
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            inputs = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
            generated_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )
            
            for generated_text in generated_texts:
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task="<ImageVQA>",
                    image_size=(
                        inputs["pixel_values"].shape[-2],
                        inputs["pixel_values"].shape[-1],
                    ),
                )
                prediction = parsed_answer['<ImageVQA>'].replace('<pad>', '').strip().lower()
                
                print(prediction)
                all_predictions.append(prediction)
                
        sample_submission['answer'] = all_predictions
        sample_submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    inference_model(test_loader, model, processor)
