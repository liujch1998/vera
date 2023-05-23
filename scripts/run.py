import argparse
from collections import defaultdict
from itertools import chain
import json
import logging
import numpy as np
import os
import random
import shutil
from tqdm import tqdm

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
import transformers
import accelerate
import wandb
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.calibration import calibration_curve

from data import FORMAT_BY_TASK, PART_BY_TASK


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def set_seed(seed=19260817, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)

def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)


accelerator = accelerate.Accelerator()
device = accelerator.device
logging.basicConfig(level=logging.INFO)
log = accelerate.logging.get_logger(__name__, log_level='INFO')
def log_info(s):
    if accelerator.is_main_process:
        log.info(s)


class DeclarativeDataset(Dataset):
    def __init__(self, args, split, tasks, tokenizer):
        self.args = args
        self.split = split
        self.tasks = tasks.split(',')
        self.tokenizer = tokenizer

        self.instances = self.load_datasets()

        if split == 'train':
            random.seed(19260817)
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self):
        instances = []
        for task_ix, task in enumerate(self.tasks):
            path = os.path.join('../data', task, f'{self.split}.json')
            with open(path) as f:
                js = json.load(f)
            if self.split == 'dev' and len(js) > 2000:
                random.seed(19260817)
                js = random.sample(js, 2000)
            for item in js:
                assert len(item['golds']) <= 1
                distractors = item['distractors'][:3] if self.split == 'train' else item['distractors']
                instance = {
                    'task': task,
                    'split': self.split,
                    'task_ix': task_ix,
                    'sources': item['golds'] + distractors,
                    'first_is_correct': 1 if len(item['golds']) > 0 else 0,
                    'is_mc': 1 if FORMAT_BY_TASK[task] == 'mc' else 0,
                }
                instances.append(instance)
            log_info(f'Loaded dataset for task {task} split {self.split} with {len(js)} instances')
        log_info(f'{self.split} set size = {len(instances)}')
        return instances

    def collate_fn(self, batch):
        task_ixs = torch.tensor([item['task_ix'] for item in batch], dtype=torch.long)
        first_is_corrects = torch.tensor([item['first_is_correct'] for item in batch], dtype=torch.long)
        is_mcs = torch.tensor([item['is_mc'] for item in batch], dtype=torch.long)

        CAP = 4 if batch[0]['split'] == 'train' else 11
        sourcess = [item['sources'] + [''] * (CAP - len(item['sources'])) for item in batch]
        sourcess_tok = [self.tokenizer.batch_encode_plus(
            sources,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_input_len)
            for sources in sourcess]
        sourcess_input_ids = torch.stack([sources_tok.input_ids for sources_tok in sourcess_tok], dim=0)
        sourcess_attention_mask = torch.stack([sources_tok.attention_mask for sources_tok in sourcess_tok], dim=0)

        result = {
            'task_ixs': task_ixs, # (B)
            'sourcess_input_ids': sourcess_input_ids, # (B, C, L)
            'sourcess_attention_mask': sourcess_attention_mask, # (B, C, L)
            'first_is_corrects': first_is_corrects, # (B)
            'is_mcs': is_mcs, # (B)
        }

        return result

class Trainer:
    def __init__(self,
                 args,
                 train_dataloader,
                 eval_dataloader,
                 tokenizer,
                 model,
                 linear,
                 optimizer,
                ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.model = model
        self.linear = linear
        self.optimizer = optimizer

        if self.args.mode == 'train':
            if not self.args.nolog and accelerator.is_main_process:
                wandb.init(project='vera', name=args.run_name, config=args)
                wandb.define_metric('train/step')
                wandb.define_metric('eval/step')
                wandb.define_metric('train/*', step_metric='train/step')
                wandb.define_metric('eval/*', step_metric='eval/step')

            self.train_sampler = iter(self.train_dataloader)

        elif self.args.mode == 'eval':
            if not self.args.nolog and accelerator.is_main_process:
                wandb.init(project='vera_eval', name=args.run_name, config=args)
                wandb.define_metric('eval/step')
                wandb.define_metric('eval/*', step_metric='eval/step')

        self.eval_accs = {}

    def logitss(self, sourcess_input_ids, sourcess_attention_mask):
        flattened_sourcess_input_ids = sourcess_input_ids.view(-1, sourcess_input_ids.size(-1)) # (B * C, L)
        flattened_sourcess_attention_mask = sourcess_attention_mask.view(-1, sourcess_attention_mask.size(-1)) # (B * C, L)
        outputs = self.model(
            input_ids=flattened_sourcess_input_ids,
            attention_mask=flattened_sourcess_attention_mask,
        )
        last_indices = flattened_sourcess_attention_mask.sum(dim=1, keepdim=True) - 1 # (B * C, 1)
        last_indices = last_indices.unsqueeze(-1).expand(-1, -1, accelerator.unwrap_model(self.model).D)
        last_hidden_state = outputs.last_hidden_state.to(device)
        hidden = last_hidden_state.gather(dim=1, index=last_indices).squeeze(1) # (B * C, D)
        flattened_logitss = self.linear(hidden).squeeze(-1) # (B * C)
        logitss = flattened_logitss.view(sourcess_input_ids.size(0), -1) # (B, C)
        # WARNING: This only works on T5 and Llama tokenizers!
        mask = (sourcess_attention_mask.sum(dim=-1) > 1) # (B, C)
        logitss[~mask] = -1e9 # If the first token is [EOS], then the source is empty
        hidden = hidden.view(sourcess_input_ids.size(0), -1, accelerator.unwrap_model(self.model).D) # (B, C, D)
        return {
            'logitss': logitss, # (B, C)
            'hidden': hidden, # (B, C, D)
            'mask': mask, # (B, C)
        }

    def loss(self, batch, split):
        B = batch['sourcess_input_ids'].size(0)
        sourcess_input_ids = batch['sourcess_input_ids']
        sourcess_attention_mask = batch['sourcess_attention_mask']
        results = self.logitss(sourcess_input_ids, sourcess_attention_mask)
        logitss = results['logitss'][:B]
        hidden = results['hidden'][:B]
        mask = results['mask'][:B]
        is_mcs = batch['is_mcs'].long()
        first_is_corrects = batch['first_is_corrects'].type(logitss.dtype)

        if self.args.loss_fn == 'margin':
            loss_fn = torch.nn.MarginRankingLoss(margin=1.0)
            losses = []
            for i in range(1, logitss.size(1)):
                losses.append(loss_fn(logitss[:, 0], logitss[:, i], torch.ones(logitss.size(0), dtype=torch.long, device=device)))
            loss_scoring = torch.stack(losses).sum()
        elif self.args.loss_fn == 'bce':
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            losses = []
            for b in range(logitss.size(0)):
                loss = []
                for i in range(1, logitss.size(1)):
                    if mask[b, i]:
                        loss.append(loss_fn(logitss[b, i], torch.zeros((), device=device)))
                if len(loss) > 0:
                    loss = torch.stack(loss).mean()
                    loss = loss + loss_fn(logitss[b, 0], first_is_corrects[b])
                else:
                    loss = loss_fn(logitss[b, 0], first_is_corrects[b])
                losses.append(loss)
            loss_scoring = torch.stack(losses).mean()
        elif self.args.loss_fn == 'mce':
            loss_fn = torch.nn.CrossEntropyLoss()
            if is_mcs.sum() > 0:
                loss_scoring = loss_fn(logitss, -100 * (1 - is_mcs))
            else:
                loss_scoring = torch.tensor(0.0, device=device)
        elif self.args.loss_fn == 'mce+bce':
            loss_fn = torch.nn.CrossEntropyLoss()
            if is_mcs.sum() > 0:
                loss_scoring = loss_fn(logitss, -100 * (1 - is_mcs))
            else:
                loss_scoring = torch.tensor(0.0, device=device)

            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            losses = []
            for b in range(logitss.size(0)):
                loss = []
                for i in range(1, logitss.size(1)):
                    if mask[b, i]:
                        loss.append(loss_fn(logitss[b, i], torch.zeros((), device=device)))
                if len(loss) > 0:
                    loss = torch.stack(loss).mean()
                    loss = loss + loss_fn(logitss[b, 0], first_is_corrects[b])
                else:
                    loss = loss_fn(logitss[b, 0], first_is_corrects[b])
                losses.append(loss)
            loss_scoring = loss_scoring + torch.stack(losses).mean()

        loss = loss_scoring

        loss_contrastive = torch.tensor(0.0, device=device)
        if self.args.contrastive_loss_type != 0:
            all_hidden = accelerator.gather(hidden) # (num_gpus * B, C, D)
            all_mask = accelerator.gather(mask) # (num_gpus * B, C)

            flattened_all_mask = all_mask.flatten(0, 1) # (num_gpus * B * C)
            true_all_mask = all_mask.detach().clone()
            true_all_mask[:, 1:] = 0
            flattened_true_all_mask = true_all_mask.flatten(0, 1) # (num_gpus * B * C)
            false_all_mask = all_mask.detach().clone()
            false_all_mask[:, 0] = 0
            flattened_false_all_mask = false_all_mask.flatten(0, 1) # (num_gpus * B * C)

            flattened_mask = mask.flatten(0, 1) # (B * C)
            true_mask = mask.detach().clone()
            true_mask[:, 1:] = 0
            flattened_true_mask = true_mask.flatten(0, 1) # (B * C)
            false_mask = mask.detach().clone()
            false_mask[:, 0] = 0
            flattened_false_mask = false_mask.flatten(0, 1) # (B * C)

            flattened_all_hidden = all_hidden.flatten(0, 1) # (num_gpus * B * C, D)
            flattened_hidden = hidden.flatten(0, 1) # (B * C, D)

            if self.args.contrastive_loss_type == 1:
                _logitss = torch.matmul(flattened_hidden, flattened_all_hidden.t()) # (B * C, num_gpus * B * C)
                _logitss = _logitss / self.args.contrastive_loss_temp
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                offset = accelerator.process_index * _logitss.size(0)
                losses = []
                for i in range(_logitss.size(0)):
                    if flattened_true_mask[i] == 1:
                        logits = _logitss[i, :].clone() # (num_gpus * B * C)
                        m = flattened_false_all_mask.detach().clone() # (num_gpus * B * C)
                        m[offset + i] = 1
                        logits[m == 0] = -1e9
                        losses.append(loss_fn(logits, torch.tensor(offset + i, dtype=torch.long, device=device)))
                    elif flattened_false_mask[i] == 1:
                        logits = _logitss[i, :].clone() # (num_gpus * B * C)
                        m = flattened_true_all_mask.detach().clone() # (num_gpus * B * C)
                        m[offset + i] = 1
                        logits[m == 0] = -1e9
                        losses.append(loss_fn(logits, torch.tensor(offset + i, dtype=torch.long, device=device)))
                loss_contrastive = torch.stack(losses).mean() # ()
            elif self.args.contrastive_loss_type == 2:
                _logitss = torch.matmul(flattened_hidden, flattened_all_hidden.t()) # (B * C, num_gpus * B * C)
                _logitss = _logitss / self.args.contrastive_loss_temp
                offset = accelerator.process_index * _logitss.size(0)
                losses = []
                for i in range(_logitss.size(0)):
                    if flattened_true_mask[i] == 1:
                        logits = _logitss[i, :].clone() # (num_gpus * B * C)
                        m = flattened_true_all_mask ^ flattened_false_all_mask # (num_gpus * B * C)
                        m[offset + i] = 0
                        logits[m == 0] = -1e9
                        logprobs = torch.nn.functional.log_softmax(logits, dim=0) # (num_gpus * B * C)
                        m = flattened_true_all_mask.detach().clone() # (num_gpus * B * C)
                        m[offset + i] = 0
                        positive_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                        losses.append(-positive_logprob)
                    elif flattened_false_mask[i] == 1:
                        logits = _logitss[i, :].clone() # (num_gpus * B * C)
                        m = flattened_true_all_mask ^ flattened_false_all_mask # (num_gpus * B * C)
                        m[offset + i] = 0
                        logits[m == 0] = -1e9
                        logprobs = torch.nn.functional.log_softmax(logits, dim=0) # (num_gpus * B * C)
                        m = flattened_false_all_mask.detach().clone() # (num_gpus * B * C)
                        m[offset + i] = 0
                        positive_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                        losses.append(-positive_logprob)
                loss_contrastive = torch.stack(losses).mean() # ()
            elif self.args.contrastive_loss_type == 3:
                cos_fn = torch.nn.CosineSimilarity(dim=1)
                _logitss = cos_fn(flattened_hidden.unsqueeze(-1), flattened_all_hidden.t().unsqueeze(0)) # (B * C, num_gpus * B * C)
                _logitss = _logitss / self.args.contrastive_loss_temp
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                offset = accelerator.process_index * _logitss.size(0)
                losses = []
                for i in range(_logitss.size(0)):
                    if flattened_true_mask[i] == 1:
                        logits = _logitss[i, :].clone() # (num_gpus * B * C)
                        m = flattened_false_all_mask.detach().clone() # (num_gpus * B * C)
                        m[offset + i] = 1
                        logits[m == 0] = -1e9
                        losses.append(loss_fn(logits, torch.tensor(offset + i, dtype=torch.long, device=device)))
                    elif flattened_false_mask[i] == 1:
                        logits = _logitss[i, :].clone() # (num_gpus * B * C)
                        m = flattened_true_all_mask.detach().clone() # (num_gpus * B * C)
                        m[offset + i] = 1
                        logits[m == 0] = -1e9
                        losses.append(loss_fn(logits, torch.tensor(offset + i, dtype=torch.long, device=device)))
                loss_contrastive = torch.stack(losses).mean() # ()
            elif self.args.contrastive_loss_type == 4:
                cos_fn = torch.nn.CosineSimilarity(dim=1)
                _logitss = cos_fn(flattened_hidden.unsqueeze(-1), flattened_all_hidden.t().unsqueeze(0)) # (B * C, num_gpus * B * C)
                _logitss = _logitss / self.args.contrastive_loss_temp
                offset = accelerator.process_index * _logitss.size(0)
                losses = []
                for i in range(_logitss.size(0)):
                    if flattened_true_mask[i] == 1:
                        logits = _logitss[i, :].clone() # (num_gpus * B * C)
                        m = flattened_true_all_mask ^ flattened_false_all_mask # (num_gpus * B * C)
                        m[offset + i] = 0
                        logits[m == 0] = -1e9
                        logprobs = torch.nn.functional.log_softmax(logits, dim=0) # (num_gpus * B * C)
                        m = flattened_true_all_mask.detach().clone() # (num_gpus * B * C)
                        m[offset + i] = 0
                        positive_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                        losses.append(-positive_logprob)
                    elif flattened_false_mask[i] == 1:
                        logits = _logitss[i, :].clone() # (num_gpus * B * C)
                        m = flattened_true_all_mask ^ flattened_false_all_mask # (num_gpus * B * C)
                        m[offset + i] = 0
                        logits[m == 0] = -1e9
                        logprobs = torch.nn.functional.log_softmax(logits, dim=0) # (num_gpus * B * C)
                        m = flattened_false_all_mask.detach().clone() # (num_gpus * B * C)
                        m[offset + i] = 0
                        positive_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                        losses.append(-positive_logprob)
                loss_contrastive = torch.stack(losses).mean() # ()

            weight = mask.sum(dim=1).float().mean().item() / all_mask.sum(dim=1).float().mean().item()
            loss_contrastive = loss_contrastive * weight
            loss = loss + self.args.contrastive_loss_coef * loss_contrastive

        scoress = torch.sigmoid(logitss) # (B, C)
        preds = logitss.argmax(dim=1) # (B)
        corrects = (preds == 0) # (B)

        return {
            'loss': loss,
            'loss_scoring': loss_scoring,
            'loss_contrastive': loss_contrastive,
            'logitss': logitss,
            'scoress': scoress,
            'preds': preds,
            'corrects': corrects,
        }

    def train(self, step):
        self.save(step=step)
        self.valid(step=step)

        accelerator.wait_for_everyone()
        self.model.train()
        self.optimizer.zero_grad()
        losses, losses_scoring, losses_contrastive = [], [], []
        corrects = []
        for _ in range(self.args.accumulate_grad_batches):
            try:
                batch = next(self.train_sampler)
            except StopIteration:
                self.train_sampler = iter(self.train_dataloader)
                batch = next(self.train_sampler)
            results = self.loss(batch, 'train')
            losses.append(results['loss'])
            losses_scoring.append(results['loss_scoring'])
            losses_contrastive.append(results['loss_contrastive'])
            corrects.append(results['corrects'])
            loss = results['loss'] / self.args.accumulate_grad_batches
            accelerator.backward(loss)
        self.optimizer.step()

        loss = torch.stack(losses).mean(dim=0, keepdim=True) # (1)
        loss_scoring = torch.stack(losses_scoring).mean(dim=0, keepdim=True) # (1)
        loss_contrastive = torch.stack(losses_contrastive).mean(dim=0, keepdim=True) # (1)
        corrects = torch.stack(corrects, dim=0) # (M, B)

        losses = accelerator.gather(loss) # (num_gpus)
        losses_scoring = accelerator.gather(loss_scoring) # (num_gpus)
        losses_contrastive = accelerator.gather(loss_contrastive) # (num_gpus)
        corrects = accelerator.gather(corrects.unsqueeze(0)) # (num_gpus, M, B)

        loss = losses.mean().item()
        loss_scoring = losses_scoring.mean().item()
        loss_contrastive = losses_contrastive.mean().item()
        acc = corrects.float().mean().item()

        if not self.args.nolog and accelerator.is_main_process:
            if step % self.args.log_interval == 0:
                wandb.log({
                    'train/step': step,
                    'train/loss': loss,
                    'train/loss_scoring': loss_scoring,
                    'train/loss_contrastive': loss_contrastive,
                    'train/acc': acc,
                })

    def valid(self, step):
        if self.args.eval_loop_cap is not None and self.args.eval_loop_cap == 0:
            return
        if step % self.args.eval_interval != 0:
            return
        if step in self.eval_accs:
            return

        stats = self.eval(step)
        eval_acc = stats['eval/mc/acc_unweighted_seen'] if 'eval/mc/acc_unweighted_seen' in stats else stats['eval/mc/acc_unweighted_all']

        if not self.args.nosave and accelerator.is_main_process:
            prev_best_step = None if len(self.eval_accs) == 0 else max(self.eval_accs, key=self.eval_accs.get)
            self.eval_accs[step] = eval_acc
            if prev_best_step is None or eval_acc > self.eval_accs[prev_best_step]:
                if prev_best_step is not None:
                    try:
                        os.remove(f'{self.args.model_dir}/ckp_{prev_best_step}.pth')
                    except:
                        log.warning(f'Cannot remove previous best ckpt!')
                shutil.copy(f'{self.args.model_dir}/last.pth', f'{self.args.model_dir}/ckp_{step}.pth')
                log_info(f'Best ckpt updated to [step {step}]')
        else:
            self.eval_accs[step] = eval_acc

    def eval(self, step):
        log_info(f'Evaluating [step {step}] ...')

        accelerator.wait_for_everyone()
        self.model.eval()

        with torch.no_grad():
            losses, losses_scoring, losses_contrastive = [], [], []
            corrects, task_ixs, first_is_corrects = [], [], []
            logitss, scoress = [], []
            for i, batch in enumerate(tqdm(self.eval_dataloader) if accelerator.is_main_process else self.eval_dataloader):
                if self.args.eval_loop_cap is not None and i == self.args.eval_loop_cap:
                    break
                results = self.loss(batch, 'eval')
                losses.append(results['loss'].detach().clone())
                losses_scoring.append(results['loss_scoring'].detach().clone())
                losses_contrastive.append(results['loss_contrastive'].detach().clone())
                corrects.append(results['corrects'].detach().clone()) # (B)
                task_ixs.append(batch['task_ixs']) # (B)
                first_is_corrects.append(batch['first_is_corrects']) # (B)
                logitss.append(results['logitss'].detach().clone()) # (B, C)
                scoress.append(results['scoress'].detach().clone()) # (B, C)

        losses = torch.stack(losses).mean(dim=0, keepdim=True) # (1)
        losses_scoring = torch.stack(losses_scoring).mean(dim=0, keepdim=True) # (1)
        losses_contrastive = torch.stack(losses_contrastive).mean(dim=0, keepdim=True) # (1)
        corrects = torch.stack(corrects, dim=0) # (M, B)
        task_ixs = torch.stack(task_ixs, dim=0) # (M, B)
        first_is_corrects = torch.stack(first_is_corrects, dim=0) # (M, B)
        logitss = torch.stack(logitss, dim=0) # (M, B, C)
        scoress = torch.stack(scoress, dim=0) # (M, B, C)

        losses = accelerator.gather(losses) # (num_gpus)
        losses_scoring = accelerator.gather(losses_scoring) # (num_gpus)
        losses_contrastive = accelerator.gather(losses_contrastive) # (num_gpus)
        corrects = accelerator.gather(corrects.unsqueeze(0)) # (num_gpus, M, B)
        task_ixs = accelerator.gather(task_ixs.unsqueeze(0)) # (num_gpus, M, B)
        first_is_corrects = accelerator.gather(first_is_corrects.unsqueeze(0)) # (num_gpus, M, B)
        logitss = accelerator.gather(logitss.unsqueeze(0)) # (num_gpus, M, B, C)
        scoress = accelerator.gather(scoress.unsqueeze(0)) # (num_gpus, M, B, C)

        # Accelerator may pad the tensors to make them divisible by the total batch size
        corrects = corrects.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N)
        task_ixs = task_ixs.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N)
        first_is_corrects = first_is_corrects.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N)
        logitss = logitss.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N, C)
        scoress = scoress.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N, C)

        loss = losses.mean().item()
        loss_scoring = losses_scoring.mean().item()
        loss_contrastive = losses_contrastive.mean().item()
        mc_corrects_by_task = defaultdict(list)
        for task_ix, correct in zip(task_ixs, corrects):
            task = self.eval_dataloader.dataset.tasks[task_ix]
            if FORMAT_BY_TASK[task] == 'mc':
                mc_corrects_by_task[task].append(correct)
        mc_corrects_by_task = {k: torch.stack(v, dim=0) for k, v in mc_corrects_by_task.items()}
        acc_by_task = {k: v.float().mean().item() for k, v in mc_corrects_by_task.items()}
        acc_unweighted_all = np.mean(list(acc_by_task.values()))
        if len([v for k, v in acc_by_task.items() if PART_BY_TASK[k] == 'seen']) > 0:
            acc_unweighted_seen = np.mean([v for k, v in acc_by_task.items() if PART_BY_TASK[k] == 'seen'])
        else:
            acc_unweighted_seen = None
        if len([v for k, v in acc_by_task.items() if PART_BY_TASK[k] == 'unseen']) > 0:
            acc_unweighted_unseen = np.mean([v for k, v in acc_by_task.items() if PART_BY_TASK[k] == 'unseen'])
        else:
            acc_unweighted_unseen = None

        stats = {
            'eval/step': step,
            'eval/loss': loss,
            'eval/loss_scoring': loss_scoring,
            'eval/loss_contrastive': loss_contrastive,
            'eval/mc/acc_unweighted_all': acc_unweighted_all,
        }
        if acc_unweighted_seen is not None:
            stats['eval/mc/acc_unweighted_seen'] = acc_unweighted_seen
        if acc_unweighted_unseen is not None:
            stats['eval/mc/acc_unweighted_unseen'] = acc_unweighted_unseen
        for task, acc in acc_by_task.items():
            stats[f'eval/mc/acc/{task}'] = acc
            if PART_BY_TASK[task] == 'seen':
                if accelerator.is_main_process:
                    print(f'{acc * 100:.2f}')

        booleans = []
        for (task_ix, first_is_correct, logits, scores) in zip(task_ixs, first_is_corrects, logitss, scoress):
            task = self.eval_dataloader.dataset.tasks[task_ix]
            if FORMAT_BY_TASK[task] == 'mc':
                assert first_is_correct.item() == 1
                booleans.append({ 'task': task, 'logit': logits[0].item(), 'score': scores[0].item(), 'label': 1.0 })
                for i in range(1, logits.size(0)):
                    if logits[i] > -1e8:
                        booleans.append({ 'task': task, 'logit': logits[i].item(), 'score': scores[i].item(), 'label': 0.0 })
            else: # boolean task
                booleans.append({ 'task': task, 'logit': logits[0].item(), 'score': scores[0].item(), 'label': first_is_correct.float().item() })
        if not self.args.nosave and accelerator.is_main_process:
            with open(f'{self.args.save_dir}/booleans_ckp{step}.json', 'w') as f:
                json.dump(booleans, f, indent=4)

        count_by_task = defaultdict(int)
        for boolean in booleans:
            count_by_task[boolean['task']] += 1
        weight_by_task = {task: 1 / count_by_task[task] for task in count_by_task}
        def compute_boolean_metrics(key):
            if key == 'all':
                bs = booleans
            elif key == 'seen':
                bs = [b for b in booleans if PART_BY_TASK[b['task']] == 'seen']
            elif key == 'unseen':
                bs = [b for b in booleans if PART_BY_TASK[b['task']] == 'unseen']
            else:
                bs = [b for b in booleans if b['task'] == key]
            if len(bs) == 0:
                return None
            weights = [weight_by_task[b['task']] for b in bs]
            labels = [int(b['label']) for b in bs]
            scores = [b['score'] for b in bs]
            logits = [b['logit'] for b in bs]

            THRESH = 0.0
            corrects = [1 if labels[i] == int(logits[i] > THRESH) else 0 for i in range(len(labels))]
            acc = np.average(corrects, weights=weights)

            fpr, tpr, thresholds = roc_curve(labels, logits, sample_weight=weights)
            auroc = auc(fpr, tpr)

            t_ap = average_precision_score(labels, logits, sample_weight=weights)
            precision, recall, thresholds = precision_recall_curve(labels, logits, sample_weight=weights)
            i = 0
            for (lo, hi) in zip(thresholds[:-1], thresholds[1:]):
                if lo <= THRESH and THRESH < hi:
                    break
                i += 1
            t_p, t_r = precision[i+1], recall[i+1]
            t_f1 = 2 * t_p * t_r / (t_p + t_r)

            f_ap = average_precision_score([1 - y for y in labels], [- y for y in logits], sample_weight=weights)
            precision, recall, thresholds = precision_recall_curve([1 - y for y in labels], [- y for y in logits], sample_weight=weights)
            i = 0
            for (lo, hi) in zip(thresholds[:-1], thresholds[1:]):
                if lo <= THRESH and THRESH < hi:
                    break
                i += 1
            f_p, f_r = precision[i+1], recall[i+1]
            f_f1 = 2 * f_p * f_r / (f_p + f_r)

            prob_true, prob_pred = calibration_curve(labels, scores, n_bins=10, strategy='quantile')
            ece = np.mean(np.abs(prob_true - prob_pred))

            return {
                'acc': acc,
                'auroc': auroc,
                't_ap': t_ap, 't_f1': t_f1, 't_p': t_p, 't_r': t_r,
                'f_ap': f_ap, 'f_f1': f_f1, 'f_p': f_p, 'f_r': f_r,
                'ece': ece,
            }
        for task in ['all', 'seen', 'unseen'] + self.eval_dataloader.dataset.tasks:
            metrics = compute_boolean_metrics(task)
            if metrics is None:
                continue
            for metric, v in metrics.items():
                stats[f'eval/bool/{task}_{metric}'] = v

        for k, v in stats.items():
            log_info(f'{k}: {v:.4f}')

        if not self.args.nolog and accelerator.is_main_process:
            wandb.log(stats)

        return stats

    def save(self, step):
        if self.args.nosave:
            return
        if step % self.args.save_interval != 0:
            return
        # this will overwrite an existing ckpt with the save filename!
        accelerator.wait_for_everyone()
        if accelerator.distributed_type == accelerate.utils.DistributedType.FSDP:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                model_state_dict = self.model.state_dict()
            with FSDP.state_dict_type(self.linear, StateDictType.FULL_STATE_DICT, save_policy):
                linear_state_dict = self.linear.state_dict()
        else:
            model_state_dict = accelerator.unwrap_model(self.model).state_dict()
            linear_state_dict = accelerator.unwrap_model(self.linear).state_dict()
        accelerator.save({
            'model': model_state_dict,
            'linear': linear_state_dict,
            'step': step,
            'eval_accs': self.eval_accs,
        }, f'{self.args.model_dir}/last.pth')
        log_info(f'[step {step}] model checkpoint saved')

def get_args():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='train or eval?')

    # common
    parser.add_argument('--model_type', type=str, default='google/t5-v1_1-small')
    parser.add_argument('--load_from_ckpt', default=None)
    parser.add_argument('--max_input_len', type=int, default=128)

    # train
    parser.add_argument('--train_tasks', type=str, default='openbookqa,arc_easy,arc_hard,ai2_science_elementary,ai2_science_middle,commonsenseqa,qasc,physical_iqa,social_iqa,winogrande_xl,com2sense_paired,sciq,quarel,quartz,cycic_mc,comve_a,csqa2,symkd_anno,gengen_anno')
    parser.add_argument('--valid_tasks', type=str, default='openbookqa,arc_easy,arc_hard,ai2_science_elementary,ai2_science_middle,commonsenseqa,qasc,physical_iqa,social_iqa,winogrande_xl,com2sense_paired,sciq,quarel,quartz,cycic_mc,comve_a,csqa2,symkd_anno,gengen_anno')
    parser.add_argument('--eval_tasks', type=str, default='openbookqa,arc_easy,arc_hard,ai2_science_elementary,ai2_science_middle,commonsenseqa,qasc,physical_iqa,social_iqa,winogrande_xl,com2sense_paired,sciq,quarel,quartz,cycic_mc,comve_a,csqa2,symkd_anno,gengen_anno,wsc273,copa,numersense,prost,spatial_cs,swag,hellaswag,codah,story_cloze_test,alphanli,strategyqa,creak,rainier_anno')
    parser.add_argument('--total_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--loss_fn', type=str, default='mce+bce', choices=['margin', 'bce', 'mce', 'mce+bce'])
    parser.add_argument('--contrastive_loss_type', type=int, default=4)
    parser.add_argument('--contrastive_loss_temp', type=float, default=0.05)
    parser.add_argument('--contrastive_loss_coef', type=float, default=0.1)

    # other
    parser.add_argument('--log_interval', type=int, default=50, help='step interval to log stats')
    parser.add_argument('--save_interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument('--eval_interval', type=int, default=1000, help='step interval to do evaluation')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--nosave', default=False, action='store_true')
    parser.add_argument('--nolog', default=False, action='store_true')
    parser.add_argument('--eval_loop_cap', type=int, default=None)

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    set_seed()

    # Set up save directories
    if not args.nosave:
        if args.mode == 'train':
            args.output_dir = '../runs/'
            args.save_dir = os.path.join(args.output_dir, args.run_name)
            args.model_dir = os.path.join(args.save_dir, 'model')
            if accelerator.is_main_process:
                for d in [args.save_dir, args.model_dir]:
                    ensure_dir(d)

        elif args.mode == 'eval':
            if args.load_from_ckpt is not None:
                args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
                args.save_dir = args.save_dir.replace('runs', 'eval')
                ckp = args.load_from_ckpt.split('ckp_')[-1].strip('.pth')
                args.save_dir += f'_ckp-{ckp}'
            else:
                log.error('You must provide --load_from_ckpt!')
                exit(-1)
            args.run_name = args.save_dir.split('/')[-1]
            if accelerator.is_main_process:
                for d in [args.save_dir]:
                    ensure_dir(d)
    
        log_info(f'Write to output directory: {args.save_dir}')
        if accelerator.is_main_process:
            with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    # Load data
    log_info(f'Loading data ...')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)
    tokenizer.max_input_len = args.max_input_len
    if 'llama' in args.model_type:
        tokenizer.pad_token = '[PAD]'
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = True
    if args.mode == 'train':
        train_dataset = DeclarativeDataset(args, 'train', args.train_tasks, tokenizer)
        eval_dataset = DeclarativeDataset(args, 'dev', args.valid_tasks, tokenizer)
        # train ds is shuffled in its constructor
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)
        train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)
    elif args.mode == 'eval':
        train_dataset = None
        eval_dataset = DeclarativeDataset(args, 'dev', args.eval_tasks, tokenizer)
        train_dataloader = None
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)
        eval_dataloader = accelerator.prepare(eval_dataloader)

    # Initialize models and optimizer
    log_info(f'Initializing models ...')
    if 't5' in args.model_type:
        model = transformers.T5EncoderModel.from_pretrained(args.model_type)
        model.D = model.shared.embedding_dim
    elif 'llama' in args.model_type:
        model = transformers.LlamaModel.from_pretrained(args.model_type)
        model.D = model.embed_tokens.embedding_dim
    linear = torch.nn.Linear(model.D, 1)
    if args.mode == 'train':
        if args.load_from_ckpt is not None:
            checkpoint = torch.load(args.load_from_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            linear.load_state_dict(checkpoint['linear'])
            checkpoint.clear()
        model, linear = accelerator.prepare(model, linear)
        optimizer = torch.optim.Adam(chain(model.parameters(), linear.parameters()), lr=args.lr)
        optimizer = accelerator.prepare(optimizer)
    elif args.mode == 'eval':
        if args.load_from_ckpt is not None:
            checkpoint = torch.load(args.load_from_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            linear.load_state_dict(checkpoint['linear'])
            step = checkpoint['step']
            checkpoint.clear()
        model, linear = accelerator.prepare(model, linear)
        optimizer = None

    # Set up trainer
    trainer = Trainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        model=model,
        linear=linear,
        optimizer=optimizer,
    )

    # Train
    if args.mode == 'train':
        steps = list(range(args.total_steps + 1))
        steps = tqdm(steps) if accelerator.is_main_process else steps
        for step in steps:
            trainer.train(step)
    elif args.mode == 'eval':
        trainer.eval(step)


if __name__ == '__main__':
    main()
