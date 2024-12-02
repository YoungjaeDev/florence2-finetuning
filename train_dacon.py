import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)
import matplotlib.pyplot as plt

from data import DaconDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model and processor
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base-ft", trust_remote_code=True, torch_dtype=torch_dtype
).to(device)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base-ft", trust_remote_code=True, torch_dtype=torch_dtype
)


def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    ).to(device)
    return inputs, answers


# Create datasets
train_dataset = DaconDataset("train")
val_dataset = DaconDataset("test")

# Create DataLoader
batch_size = 16
num_workers = 64

# Epoch
epochs = 3

# Log interval
log_interval = 200

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    collate_fn=collate_fn, 
    num_workers=num_workers,
    pin_memory=True
)

def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("===== Model Information =====")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {train_loader.batch_size}")
    print(f"Number of epochs: {epochs}")
    print("============================\n")

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Best model tracking
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf')
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(progress_bar):
            inputs, answers = batch

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"].to(torch_dtype)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss

            # Print example predictions every log_interval steps
            if i % log_interval == 0:
                print(f"\nStep {i} Loss: {loss.item():.4f}")
                
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
                generated_texts = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )

                for generated_text, answer in zip(generated_texts, answers):
                    # task_answer_post_processing_type == 'pure_text'
                    parsed_answer = processor.post_process_generation(
                        generated_text,
                        task="<ImageVQA>",
                        image_size=(
                            inputs["pixel_values"].shape[-2],
                            inputs["pixel_values"].shape[-1],
                        ),
                    )
                    print("Example prediction:")
                    print(f"Ground Truth: {answers[0]}")
                    print(f"Prediction: {parsed_answer['<ImageVQA>']}\n")
                    
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f"\nEpoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                inputs, answers = batch

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"].to(torch_dtype)
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        print(f"Epoch {epoch + 1} Average Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            history['best_val_loss'] = best_val_loss
            
            best_model_dir = "./model_checkpoints/best_model"
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            processor.save_pretrained(best_model_dir)
            
            print(f"\nNew best model saved! (Validation Loss: {best_val_loss:.4f})")

        # Save regular checkpoint
        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
    print("\n===== Training Complete =====")
    print(f"Best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label='Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('./model_checkpoints/training_history.png')
    plt.close()

    return history


# for param in model.vision_tower.parameters():
#   param.is_trainable = False
  
history = train_model(train_loader, val_loader, model, processor, epochs=epochs)
