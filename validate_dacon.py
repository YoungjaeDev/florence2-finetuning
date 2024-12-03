import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from data import DaconVQADataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Hugging Face 로그인
def init_hf_login():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")  # 환경 변수에서 토큰을 가져오거나
    if token is None:
        raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in the environment variables.")
    login(token=token)
    
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
model_dir = "./model_checkpoints/best_model"
model_name = "microsoft/Florence-2-base-ft"
model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, config=model_config).to(device)
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    )
    return inputs, answers


# Create datasets
dataset = DaconVQADataset("train")
# val_dataset = DaconVQADataset("test")

# split dataset
random_seed = 42
generator = torch.Generator().manual_seed(random_seed)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)


# Create validation dataset and DataLoader
val_loader = DataLoader(
    val_dataset, 
    batch_size=16, 
    collate_fn=collate_fn, 
    num_workers=4,
    pin_memory=True,
)

def validate_model(val_loader, model, processor):
    # Push to Hub
    try:
        print("Pushing model and processor to Hugging Face Hub...")
        model.push_to_hub("YoungjaeDev/DaconVQA-Florence2-ft-base")
        processor.push_to_hub("YoungjaeDev/DaconVQA-Florence2-ft-base")
        print("Successfully pushed to Hub!")
    except Exception as e:
        print(f"Error pushing to Hub: {str(e)}")
        
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs, answers = batch
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
            
            for generated_text, answer in zip(generated_texts, answers):
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task="<ImageVQA>",
                    image_size=(
                        inputs["pixel_values"].shape[-2],
                        inputs["pixel_values"].shape[-1],
                    ),
                )
                prediction = parsed_answer['<ImageVQA>'].replace('<pad>', '').strip().lower()
                ground_truth = answer.lower().strip()
                
                if prediction == ground_truth:
                    total_correct += 1
                total_samples += 1

    accuracy = total_correct / total_samples * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
    
        
if __name__ == "__main__":
    
    init_hf_login()
    validate_model(val_loader, model, processor)
