import argparse
import os
from functools import partial

import friendlywords as fw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)

import wandb
from data import DaconVQADataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score

RESUME = False
model_checkpoint_dir = "./model_checkpoints/ddp_best_model"

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    )
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    return inputs, answers


def train_model(rank, world_size, batch_size=32, use_lora=False, epochs=5, lr=1e-6, eval_steps=200, run_name=None):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    # wandb 초기화 (rank 0만)
    if rank == 0:
        wandb.init(project="DaconVQA", name=run_name)
        wandb.config.update({
            "batch_size": batch_size,
            "use_lora": use_lora,
            "epochs": epochs,
            "learning_rate": lr,
            "eval_steps": eval_steps,
            "world_size": world_size,
        })

    # 모델과 프로세서 로드
    model_config = AutoConfig.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base-ft" if not RESUME else model_checkpoint_dir, trust_remote_code=True, config=model_config
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base-ft", trust_remote_code=True
    )

    if use_lora:
        TARGET_MODULES = [
            "q_proj", "o_proj", "k_proj", "v_proj",
            "linear", "Conv2d", "lm_head", "fc2"
        ]

        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, config)

    # 이미지 인코더 파라미터 고정
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    # DDP 모델 설정
    model = DDP(model, device_ids=[rank])

    # 데이터셋 및 데이터로더 생성
    dataset = DaconVQADataset("train")
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=8,
        sampler=train_sampler,
        # pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=8,
        sampler=val_sampler,
        # pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0

    # 학습 루프
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank)
        for i, batch in enumerate(progress_bar):
            inputs, answers = batch
            
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            outputs = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                labels=labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            global_step += 1

            if global_step % eval_steps == 0:
                # Log training metrics
                if rank == 0:
                    avg_train_loss = train_loss / global_step
                    wandb.log({
                        "train_loss": avg_train_loss,
                        "step": global_step
                    })

        # 검증 단계
        model.eval()
        val_loss = 0
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}", position=rank):
                inputs, answers = batch
                
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                outputs = model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()

                # Generate predictions
                if rank == 0:
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
                        
                        all_predictions.append(prediction)
                        all_ground_truths.append(ground_truth)

        # 전체 검증 손실 계산 및 기록 (rank 0만)
        avg_val_loss = val_loss / len(val_loader)
        if rank == 0:
            accuracy = accuracy_score(all_ground_truths, all_predictions)
            wandb.log({
                "val_loss": avg_val_loss,
                "val_accuracy": accuracy,
                "epoch": epoch + 1
            })
            print(f"Epoch {epoch + 1} Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Epoch {epoch + 1} Validation Accuracy: {accuracy:.4f}")

            # 모델 저장
        # 모델 저장 (rank 0만)
        if rank == 0:
            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_dir = f"./model_checkpoints/{run_name}/best_model"
                os.makedirs(best_model_dir, exist_ok=True)
                model.module.save_pretrained(best_model_dir)
                processor.save_pretrained(best_model_dir)

    cleanup()

def main():
    parser = argparse.ArgumentParser(description="Train Florence-2 model on DaconVQA dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args.batch_size, args.epochs, args.lr, args.eval_steps, args.run_name),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
