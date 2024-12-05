import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from functools import partial
from datetime import timedelta

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
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv()

import wandb

from data import DaconVQADataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score
from utils.config import Config

def setup(rank, world_size):
    print(f"Setting up process group for rank {rank} with world size {world_size} ... ")
    port = int(os.environ.get("MASTER_PORT", "29533"))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=10))
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def collate_fn(batch, processor):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    )
    return inputs, answers


def train_model(rank, world_size, config: Config, run_name=None):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    # wandb 초기화 (rank 0만)
    if rank == 0:
        # wandb.init(project="DaconVQA", 
        #            name=run_name, 
        #            settings=wandb.Settings(init_timeout=450),
        #            reinit=True,  # Allow reinit if needed
        #            resume="allow"  # Allow resuming if a run crashes   
        #         )
        # wandb.config.update(config.__dict__)

        print(f"Process {rank} started.")


    # 모델과 프로세서 로드
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name if not config.training.resume else config.training.resume_path,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=getattr(torch, config.model.dtype)
    )
    model = model.to(rank)
    processor = AutoProcessor.from_pretrained(
        config.model.name, trust_remote_code=config.model.trust_remote_code
    )
        
    if config.model.use_lora:
        TARGET_MODULES = [
            "q_proj", "o_proj", "k_proj", "v_proj",
            "linear", "Conv2d", "lm_head", "fc2"
        ]

        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=config.lora.dropout,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)

    # 이미지 인코더 파라미터 고정
    if not config.model.train_vision_encoder:
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    # DDP 모델 설정
    model = DDP(model, device_ids=[rank])

    # 데이터셋 및 데이터로더 생성
    dataset = DaconVQADataset("train")
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=config.training.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        prefetch_factor=config.training.prefetch_factor
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=config.training.num_workers,
        sampler=val_sampler,
        pin_memory=True,
        prefetch_factor=config.training.prefetch_factor
    )

    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
    num_training_steps = config.training.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0

    # 학습 루프
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': []}
    
    if rank == 0:
        csv_path = f"./model_checkpoints/{run_name}/training_history.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("epoch,train_loss,val_loss\n")
            
    for epoch in range(config.training.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"[RANK {rank}] Training Epoch {epoch + 1}/{config.training.epochs}", position=rank)
        for i, batch in enumerate(progress_bar):
            inputs, answers = batch
            
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            outputs = model(
                inputs["input_ids"].to(device),  # input_ids는 LongTensor 유지
                inputs["pixel_values"].to(device, dtype=getattr(torch, config.model.dtype)),  # pixel_values는 config.model.dtype 사용
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

            if global_step % config.training.log_steps == 0:
                # Log training metrics
                if rank == 0:
                    avg_train_loss = train_loss / global_step
                    # wandb.log({
                    #     "train_loss": avg_train_loss,
                    #     "step": global_step
                    # })


        # 각 에폭 끝에서 평균 train_loss 계산
        # avg_train_loss = train_loss / len(train_loader)

        train_loss = torch.tensor(train_loss, device=device)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss / (len(train_loader) * world_size)

        print(f"Rank {rank}: Epoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")
        history['train_loss'].append(avg_train_loss)

        # 검증 단계
        model.eval()
        val_loss = 0
        # val_steps = 0
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[RANK {rank}] Validation Epoch {epoch + 1}/{config.training.epochs}", position=rank):
                inputs, answers = batch
            
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                    # truncation=True,
                    # max_length=800
                ).input_ids.to(device)

                outputs = model(
                    inputs["input_ids"].to(device),  # input_ids는 LongTensor 유지
                    inputs["pixel_values"].to(device, dtype=getattr(torch, config.model.dtype)),  # pixel_values는 config.model.dtype 사용
                    labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()
                # val_steps += 1

                # Generate predictions
                # if rank == 0:
                #     # print(f"input_ids shape: {inputs['input_ids'].shape}")
                #     # print(f"pixel_values shape: {inputs['pixel_values'].shape}")
                    
                #     generated_ids = model.module.generate(
                #         inputs["input_ids"].to(device),  # input_ids는 LongTensor 유지
                #         inputs["pixel_values"].to(device, dtype=getattr(torch, config.model.dtype)),  # pixel_values는 config.model.dtype 사용
                #         max_new_tokens=1024,
                #         num_beams=3,
                #     )
                #     generated_texts = processor.batch_decode(
                #         generated_ids, skip_special_tokens=False
                #     )
                    
                #     for generated_text, answer in zip(generated_texts, answers):
                #         parsed_answer = processor.post_process_generation(
                #             generated_text,
                #             task="<ImageVQA>",
                #             image_size=(
                #                 inputs["pixel_values"].shape[-2],
                #                 inputs["pixel_values"].shape[-1],
                #             ),
                #         )
                #         prediction = parsed_answer['<ImageVQA>'].replace('<pad>', '').strip().lower()
                #         ground_truth = answer.lower().strip()
                        
                #         all_predictions.append(prediction)
                #         all_ground_truths.append(ground_truth)
        
        # 모든 프로세스의 val_loss 동기화
        val_loss = torch.tensor(val_loss, device=device)
        # val_steps = torch.tensor(val_steps, device=device)
        
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        # dist.all_reduce(val_steps, op=dist.ReduceOp.SUM)
        
        # 전체 검증 손실 계산 (모든 프로세스에서 동일한 값을 가짐)
        # avg_val_loss = (val_loss / val_steps).item()
        avg_val_loss = val_loss / (len(val_loader) * world_size) 
        history['val_loss'].append(avg_val_loss)
        
        if rank == 0:
            # accuracy = accuracy_score(all_ground_truths, all_predictions)
            # wandb.log({
            #     "val_loss": avg_val_loss,
            #     "val_accuracy": accuracy, # only rank 0
            #     "epoch": epoch + 1
            # })
            print(f"All processes: Epoch {epoch + 1} Average Validation Loss: {avg_val_loss:.4f}")
            # print(f"Rank {rank}: Epoch {epoch + 1} Validation Accuracy: {accuracy:.4f}")
            
            csv_path = f"./model_checkpoints/{run_name}/training_history.csv"
            with open(csv_path, 'a') as f:
                f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")

        # 모델 저장 (rank 0만)
        if rank == 0:
            # avg_val_loss = val_loss / len(val_loader)
            avg_val_loss = val_loss / (len(val_loader) * world_size)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_dir = f"./model_checkpoints/{run_name}/best_model"
                os.makedirs(best_model_dir, exist_ok=True)
                model.module.save_pretrained(best_model_dir)
                processor.save_pretrained(best_model_dir)
    

    # 학습 종료 후 그래프 생성 (rank 0만)
    if rank == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.axvline(x=best_epoch-1, color='r', linestyle='--', label='Best Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig(f'./model_checkpoints/{run_name}/training_history.png')
        plt.close()
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description="Train Florence-2 model on DaconVQA dataset")
    parser.add_argument("--config", type=str, default="config/train_config.yaml", help="Path to config file")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)
    
    # wandb 초기화는 메인 프로세스에서만 한 번 수행
    # wandb.login(key=os.getenv("WANDB_API_KEY"))
    
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    mp.spawn(
        train_model,
        args=(world_size, config, args.run_name),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
