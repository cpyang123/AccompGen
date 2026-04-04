import os
import gc
import time
import math
import json
import wandb
import torch
import random
import numpy as np
from abctoolkit.transpose import Key2index, Key2Mode
from utils import *
from config import *
from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, LlamaConfig, get_scheduler, get_constant_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

Index2Key = {index: key for key, index in Key2index.items() if index not in [1, 11]}
Mode2Key = {mode: key for key, mode_list in Key2Mode.items() for mode in mode_list }

# Set up distributed training
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

if world_size > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl') if world_size > 1 else None
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set random seed
seed = 0 + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


batch_size = BATCH_SIZE

patchilizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                    max_length=PATCH_LENGTH, 
                    max_position_embeddings=PATCH_LENGTH,
                    n_embd=HIDDEN_SIZE,
                    num_attention_heads=HIDDEN_SIZE//64,
                    vocab_size=1)
char_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS, 
                            max_length=PATCH_SIZE+1, 
                            max_position_embeddings=PATCH_SIZE+1,
                            hidden_size=HIDDEN_SIZE,
                            num_attention_heads=HIDDEN_SIZE//64,
                            vocab_size=128)

model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=char_config)

model = model.to(device)

# print parameter number
print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

scaler = GradScaler()
is_autocast = True
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def clear_unused_tensors():
    gc.disable()  # Temporarily disable garbage collection
    try:
        # Get the set of tensor ids used by the model
        if hasattr(model, "module"):
            model_tensors = {id(p) for p in model.module.parameters()}
        else:
            model_tensors = {id(p) for p in model.parameters()}
        
        # Get the set of tensor ids used by the optimizer
        optimizer_tensors = {
            id(state) 
            for state_dict in optimizer.state.values() 
            for state in state_dict.values()
            if isinstance(state, torch.Tensor)  # Ensure only tensors are considered
        }

        # List of all CUDA tensors currently in memory
        tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]
        
        # Create weak references to avoid interfering with garbage collection
        tensor_refs = [weakref.ref(tensor) for tensor in tensors]

        for tensor_ref in tensor_refs:
            tensor = tensor_ref()  # Dereference the weak reference
            if tensor is not None and id(tensor) not in model_tensors and id(tensor) not in optimizer_tensors:
                # Mark the tensor for deletion
                tensor.detach_()  # Detach from computation graph
                del tensor  # Delete the tensor reference
    except:
        pass

    finally:
        gc.enable()  # Re-enable garbage collection
        gc.collect()  # Force a garbage collection
        torch.cuda.empty_cache()  # Clear the CUDA cache

def collate_batch(input_batches):
    
    input_patches, input_masks = zip(*input_batches)
    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=0)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True, padding_value=0)

    return input_patches.to(device), input_masks.to(device)

def split_into_minibatches(input_patches, input_masks, minibatch_size):
    minibatches = []
    for start_idx in range(0, len(input_patches), minibatch_size):
        end_idx = start_idx + minibatch_size
        minibatch_patches = input_patches[start_idx:end_idx]
        minibatch_masks = input_masks[start_idx:end_idx]
        minibatches.append((minibatch_patches, minibatch_masks))
    return minibatches

class NotaGenDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filepath = self.filenames[idx]['path']
        ori_key = Mode2Key[self.filenames[idx]['key']]

        # choose a key to transpose, according to a probility distribution
        ori_key_index = Key2index[ori_key]
        available_index = [(ori_key_index + offset) % 12 for offset in range(-3, 4)]
        index_prob = [1/16, 2/16, 3/16, 4/16, 3/16, 2/16, 1/16]
        index_prob_range = [0] + [sum(index_prob[0 : i + 1]) for i in range(len(index_prob))]
        random_number = random.random()
        for i in range(len(index_prob_range) - 1):
            if index_prob_range[i] <= random_number < index_prob_range[i + 1]:
                des_key_index = available_index[i]
        if des_key_index == 1:
            des_key = 'Db' if random.random() < 0.8 else 'C#'   
        elif des_key_index == 11:
            des_key = 'B' if random.random() < 0.8 else 'Cb'
        elif des_key_index == 6:
            des_key = 'F#' if random.random() < 0.5 else 'Gb'
        else:
            des_key = Index2Key[des_key_index]

        folder = os.path.dirname(filepath)
        name = os.path.split(filepath)[-1]
        des_filepath = os.path.join(folder, des_key, name + '_' + des_key + '.abc')
        
        with open(des_filepath, 'r', encoding='utf-8') as f:
            abc_text = f.read()

        file_bytes = patchilizer.encode_train(abc_text)
        file_masks = [1] * len(file_bytes)

        file_bytes = torch.tensor(file_bytes, dtype=torch.long)
        file_masks = torch.tensor(file_masks, dtype=torch.long)
        
        return file_bytes, file_masks


def process_one_batch(batch):
    input_patches, input_masks = batch
    loss = model(input_patches, input_masks).loss

    # Reduce the loss on GPU 0
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)

    return loss


# do one epoch for training
def train_epoch(epoch, train_loader, global_step_offset=0):
    tqdm_train_set = tqdm(train_loader)
    total_train_loss = 0
    iter_idx = 1
    model.train()
    train_steps = global_step_offset + (epoch - 1) * len(train_loader)

    for batch in tqdm_train_set:
        minibatches = split_into_minibatches(batch[0], batch[1], BATCH_SIZE//ACCUMULATION_STEPS)
        for minibatch in minibatches:
            with autocast():
                loss = process_one_batch(minibatch) / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            total_train_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        tqdm_train_set.set_postfix({str(global_rank)+'_train_loss': total_train_loss / iter_idx})
        train_steps += 1

        # Log the training loss to wandb
        if global_rank==0 and WANDB_LOGGING:
            wandb.log({"train_loss": total_train_loss / iter_idx}, step=train_steps)

        iter_idx += 1
        if iter_idx % 1000 == 0:
            clear_unused_tensors()

    return total_train_loss / (iter_idx-1)


def make_dataloader(files):
    batch_nums = int(len(files) / batch_size)
    files = files[:batch_nums * batch_size]
    dataset = NotaGenDataset(files)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch,
                        sampler=sampler, shuffle=(sampler is None))
    return loader, sampler

# do one epoch for eval
def eval_epoch(eval_loader):
    tqdm_eval_set = tqdm(eval_loader)
    total_eval_loss = 0
    total_eval_bpb = 0
    iter_idx = 1
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm_eval_set:
        minibatches = split_into_minibatches(batch[0], batch[1], BATCH_SIZE//ACCUMULATION_STEPS)
        for minibatch in minibatches:
            with torch.no_grad():
                loss = process_one_batch(minibatch) / ACCUMULATION_STEPS
            total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({str(global_rank)+'_eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1
    return total_eval_loss / (iter_idx-1)

# train and eval
if __name__ == "__main__":

    # Initialize wandb
    if WANDB_LOGGING and global_rank==0:
        wandb.login(key=WANDB_KEY)
        wandb.init(project="notagen",
                   entity="SchenkerDiff",
                   name=WANDB_NAME)

    # load data
    with open(DATA_TRAIN_INDEX_PATH, "r", encoding="utf-8") as f:
        print("Loading Data...")
        train_files = []
        for line in f:
            train_files.append(json.loads(line))

    with open(DATA_EVAL_INDEX_PATH, "r", encoding="utf-8") as f:
        print("Loading Data...")
        eval_files = []
        for line in f:
            eval_files.append(json.loads(line))

    if len(eval_files) == 0:
        train_files, eval_files = split_data(train_files)

    # Split train files into synthetic and real
    synthetic_train_files = [f for f in train_files if "synthetic" in f["path"]]
    real_train_files      = [f for f in train_files if "synthetic" not in f["path"]]
    print(f"Curriculum split — synthetic: {len(synthetic_train_files)}, real: {len(real_train_files)}")

    random.shuffle(synthetic_train_files)
    random.shuffle(real_train_files)
    random.shuffle(eval_files)

    eval_set, eval_sampler = make_dataloader(eval_files)

    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if not LOAD_FROM_CHECKPOINT:
        if os.path.exists(PRETRAINED_PATH):
            # Load pre-trained checkpoint to CPU
            checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')

            # Load state dict to CPU model first, then move the model to GPU
            if torch.cuda.device_count() > 1:
                cpu_model = deepcopy(model.module)
                cpu_model.load_state_dict(checkpoint['model'])
                model.module.load_state_dict(cpu_model.state_dict())
            else:
                cpu_model = deepcopy(model)
                cpu_model.load_state_dict(checkpoint['model'])
                model.load_state_dict(cpu_model.state_dict())

            print(f"Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Loss {checkpoint['min_eval_loss']}")

            pre_epoch = 0
            best_epoch = 0
            min_eval_loss = 100
        else:
            raise Exception('Pre-trained Checkpoint not found. Please check your pre-trained ckpt path.')

    else:
        if os.path.exists(WEIGHTS_PATH):
            # Load checkpoint to CPU
            checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')

            if torch.cuda.device_count() > 1:
                cpu_model = deepcopy(model.module)
                cpu_model.load_state_dict(checkpoint['model'])
                model.module.load_state_dict(cpu_model.state_dict())
            else:
                cpu_model = deepcopy(model)
                cpu_model.load_state_dict(checkpoint['model'])
                model.load_state_dict(cpu_model.state_dict())
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_sched'])
            pre_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            min_eval_loss = checkpoint['min_eval_loss']
            print("Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)
            checkpoint = None

        else:
            raise Exception('Checkpoint not found to continue training. Please check your parameter settings.')


    def run_phase(phase_name, phase_files, num_epochs, epoch_offset, global_step_offset, best_epoch, min_eval_loss):
        train_set, train_sampler = make_dataloader(phase_files)
        print(f"\n{'='*20} {phase_name} phase ({num_epochs} epochs) {'='*20}")
        steps_this_phase = num_epochs * len(train_set)
        for epoch in range(1, num_epochs + 1):
            global_epoch = epoch + epoch_offset
            train_sampler.set_epoch(global_epoch)
            eval_sampler.set_epoch(global_epoch)
            print('-' * 21 + f"Epoch {global_epoch} ({phase_name})" + '-' * 21)
            train_loss = train_epoch(epoch, train_set, global_step_offset=global_step_offset)
            eval_loss = eval_epoch(eval_set)
            if global_rank == 0:
                if WANDB_LOGGING:
                    wandb.log({"eval_loss_epoch": eval_loss, "epoch": global_epoch,
                               "phase": phase_name})
                with open(LOGS_PATH, 'a') as f:
                    f.write(f"Epoch {global_epoch} [{phase_name}]\ntrain_loss: {train_loss}\neval_loss: {eval_loss}\ntime: {time.asctime(time.localtime(time.time()))}\n\n")
                if eval_loss < min_eval_loss:
                    best_epoch = global_epoch
                    min_eval_loss = eval_loss
                    ckpt = {
                        'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_sched': lr_scheduler.state_dict(),
                        'epoch': global_epoch,
                        'best_epoch': best_epoch,
                        'min_eval_loss': min_eval_loss,
                        'phase': phase_name,
                    }
                    torch.save(ckpt, WEIGHTS_PATH)
            if world_size > 1:
                dist.barrier()
        return best_epoch, min_eval_loss, steps_this_phase

    # Phase 1: synthetic data
    best_epoch, min_eval_loss, phase1_steps = run_phase(
        "synthetic", synthetic_train_files, NUM_EPOCHS_SYNTHETIC,
        epoch_offset=pre_epoch, global_step_offset=0,
        best_epoch=best_epoch, min_eval_loss=min_eval_loss)
    # Phase 2: real data (step offset continues from end of phase 1)
    best_epoch, min_eval_loss, _ = run_phase(
        "real", real_train_files, NUM_EPOCHS_REAL,
        epoch_offset=pre_epoch + NUM_EPOCHS_SYNTHETIC, global_step_offset=phase1_steps,
        best_epoch=best_epoch, min_eval_loss=min_eval_loss)

    if global_rank == 0:
        print("Best Eval Epoch : " + str(best_epoch))
        print("Min Eval Loss : " + str(min_eval_loss))
