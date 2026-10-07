"""Project 4: Flow Matching Training

含 TODO 16: 实现 Rectified Flow loss
"""
import argparse
import csv
import json
import os
import time
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    """根据 config 构建 model（UNet 或 DiT）。

    约定：两个 backbone 的 `num_classes` 都传**真实类别数**（CIFAR-10 是 10），
    null token 由模型内部多分配一个 embedding，索引固定为 num_classes。
    """
    m = cfg['model']
    if m['type'] == 'unet':
        from model.unet import ConditionalUNet
        return ConditionalUNet(
            in_channels=3,
            out_channels=3,
            base_channels=m['base_channels'],
            num_classes=m['num_classes'],
        )
    elif m['type'] == 'dit':
        from model.dit import SimpleDiT
        return SimpleDiT(
            img_size=32,
            patch_size=m.get('patch_size', 2),
            in_ch=3,
            embed_dim=m['embed_dim'],
            depth=m['depth'],
            num_heads=m['num_heads'],
            num_classes=m['num_classes'],
            time_scale=m.get('time_scale', 1.0),
        )
    else:
        raise ValueError(f"未知的 model.type: {m['type']}（只支持 'unet' / 'dit'）")


def compute_fm_loss(model, x_1, y, cond_drop_prob, num_classes):
    """
    ============================================================
    TODO 16: 实现 Rectified Flow / Conditional Flow Matching loss
    ============================================================

    输入:
        model: 接受 (x, t, y) 输出 velocity 预测 v_pred (B, 3, 32, 32)
        x_1: 真实图像 (B, 3, 32, 32)
        y: 类别 label (B,) ∈ {0, ..., num_classes-1}
        cond_drop_prob: 训练时 y 替换为 null 的概率（CFG）
        num_classes: 总类别数（最后一个 index 用作 null）

    要求:
        1. 采样 t ~ U[0, 1]，shape = (B,)
        2. 采样 epsilon ~ N(0, I)，shape = x_1.shape
        3. 构造 x_t = (1 - t) * epsilon + t * x_1
        4. Target = x_1 - epsilon
        5. 做 conditional dropout: 以 cond_drop_prob 概率把 y 替换为 num_classes（null token）
        6. 调用 model(x_t, t, y) → v_pred
        7. 返回 MSE(v_pred, target)

    Hints:
        - t.view(-1, 1, 1, 1) 让 t 能与 (B, C, H, W) broadcast
        - y_null = torch.full_like(y, num_classes)
        - drop_mask = torch.rand(B) < cond_drop_prob
        - y_in = torch.where(drop_mask, y_null, y)
    ============================================================
    """
    B = x_1.shape[0]
    device = x_1.device

    # Sample a point on the conditional linear probability path.
    t = torch.rand(B, device=device)
    epsilon = torch.randn_like(x_1)
    t_view = t.view(B, 1, 1, 1)
    x_t = (1.0 - t_view) * epsilon + t_view * x_1
    target = x_1 - epsilon

    # The final embedding entry is reserved for the unconditional/null token.
    y_null = torch.full_like(y, num_classes)
    drop_mask = torch.rand(B, device=device) < cond_drop_prob
    y_in = torch.where(drop_mask, y_null, y)

    velocity = model(x_t, t, y_in)
    return F.mse_loss(velocity, target)


def _lr_lambda(step, train_cfg):
    warmup = int(train_cfg.get('warmup_steps', 0))
    decay_start = int(train_cfg.get('decay_start_step', train_cfg['max_steps']))
    max_steps = int(train_cfg['max_steps'])
    floor = float(train_cfg.get('min_lr_ratio', 0.1))
    if warmup and step < warmup:
        return max(1e-8, (step + 1) / warmup)
    if step < decay_start:
        return 1.0
    progress = min(1.0, max(0.0, (step - decay_start) / max(1, max_steps - decay_start)))
    return floor + (1.0 - floor) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def _save_checkpoint(path, checkpoint):
    temporary = path + '.tmp'
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def train(cfg, output_dir, seed=42, precision='fp32', resume=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Data
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # to [-1, 1]
    ])
    ds = datasets.CIFAR10(cfg['data']['root'], train=True, download=True, transform=tf)
    data_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(cfg['data']['num_workers'] > 0),
        generator=data_generator,
    )

    model = build_model(cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    amp_enabled = precision == 'bf16' and device.type == 'cuda'
    if precision not in ('fp32', 'bf16'):
        raise ValueError("precision must be 'fp32' or 'bf16'")
    print(f"device={device} precision={precision} seed={seed}")

    ema_decays = cfg['train'].get('ema_decays', [cfg['train'].get('ema_decay', 0.9999)])
    ema_states = [{k: v.clone().detach() for k, v in model.state_dict().items()}
                  for _ in ema_decays]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: _lr_lambda(s, cfg['train']))

    step = 0
    losses = []
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt:
            if isinstance(ckpt['ema'], dict) and 'models' in ckpt['ema']:
                saved_decays, saved_models = ckpt['ema']['decays'], ckpt['ema']['models']
                for i, decay in enumerate(ema_decays):
                    match = next((j for j, old in enumerate(saved_decays)
                                  if abs(float(old) - float(decay)) < 1e-8), None)
                    if match is not None:
                        ema_states[i] = {k: v.to(device).clone().detach()
                                         for k, v in saved_models[match].items()}
            else:
                # Backward compatibility with Project4 v1's single EMA state dict.
                ema_states = [{k: v.to(device).clone().detach()
                               for k, v in ckpt['ema'].items()} for _ in ema_decays]
        if 'opt' in ckpt:
            opt.load_state_dict(ckpt['opt'])
        step = int(ckpt.get('step', 0))
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'rng' in ckpt:
            rng = ckpt['rng']
            torch.set_rng_state(rng['torch'].cpu())
            random.setstate(rng['python'])
            data_generator.set_state(rng['data_generator'].cpu())
            if torch.cuda.is_available() and rng.get('cuda') is not None:
                torch.cuda.set_rng_state_all(rng['cuda'])
        print(f"Resumed from {resume} at step {step}")

    initial_step = step
    num_classes = cfg['model']['num_classes']
    cond_drop_prob = cfg['train'].get('cond_drop_prob', 0.1)
    t0 = time.time()
    prior_elapsed = float(ckpt.get('elapsed_seconds', 0.0)) if resume else 0.0
    log_path = os.path.join(output_dir, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerow(['step', 'loss_100', 'lr', 'step_per_sec', 'elapsed_seconds'])

    while step < cfg['train']['max_steps']:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                loss = compute_fm_loss(model, x, y, cond_drop_prob, num_classes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train'].get('grad_clip', 1.0))
            opt.step()
            scheduler.step()

            with torch.no_grad():
                current_state = model.state_dict()
                for ema_decay, ema in zip(ema_decays, ema_states):
                    for k, v in current_state.items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)

            losses.append(loss.item())
            step += 1

            if step % cfg['train'].get('log_every', 100) == 0:
                mean = sum(losses[-100:]) / min(len(losses), 100)
                elapsed = time.time() - t0
                speed = (step - initial_step) / max(elapsed, 1e-6)
                lr = opt.param_groups[0]['lr']
                total_elapsed = prior_elapsed + elapsed
                print(f"step {step:6d}/{cfg['train']['max_steps']} | loss {mean:.4f} | "
                      f"lr {lr:.2e} | {speed:.2f} step/s | elapsed {total_elapsed/3600:.2f}h", flush=True)
                with open(log_path, 'a', newline='', encoding='utf-8') as stream:
                    csv.writer(stream).writerow([step, f'{mean:.7f}', f'{lr:.9g}',
                                                 f'{speed:.5f}', f'{total_elapsed:.1f}'])

            if step % cfg['train'].get('save_every', 10000) == 0 or step >= cfg['train']['max_steps']:
                elapsed_total = prior_elapsed + time.time() - t0
                ckpt = {
                    'step': step,
                    'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'ema': {'decays': list(ema_decays),
                            'models': [{k: v.detach().cpu() for k, v in ema.items()}
                                       for ema in ema_states]},
                    'opt': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'cfg': cfg,
                    'seed': seed,
                    'precision': precision,
                    'elapsed_seconds': elapsed_total,
                    'rng': {'torch': torch.get_rng_state(),
                            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            'python': random.getstate(),
                            'data_generator': data_generator.get_state()},
                }
                _save_checkpoint(os.path.join(output_dir, 'latest.pt'), ckpt)
                snapshot_every = int(cfg['train'].get('snapshot_every', 50000))
                if step % snapshot_every == 0 or step >= cfg['train']['max_steps']:
                    _save_checkpoint(os.path.join(output_dir, f'step_{step}.pt'), ckpt)
                print(f"saved checkpoint at step {step}", flush=True)

            if step >= cfg['train']['max_steps']:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--resume', default=None, help='checkpoint path to resume from')
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.output, seed=args.seed, precision=args.precision, resume=args.resume)
