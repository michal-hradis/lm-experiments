import argparse
import logging
import os
import torch
import json
import typing
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from diff_print import console_transcription_errors
from autoregressive_sampling import sample_characters
import  diff_print

    
class LearningLoop:
    @staticmethod
    def add_params(parser: argparse.ArgumentParser):
        parser.add_argument('--name', required=True, help='Name for tensorboard.')
        parser.add_argument('--tensorboard-path', default='./tb_log/', help='Path for tensorboard logs.')
        parser.add_argument('-g', '--gpu-id', type=int,
                            help="If not set setGPU() is called. Set this to 0 on desktop. Leave it empty on SGE.")
        parser.add_argument('--start-iteration', default=0, type=int)
        parser.add_argument('--max-iterations', default=500000, type=int)
        parser.add_argument('--view-step', default=500, type=int)

        parser.add_argument('--optimization-config', default='{"type":"Adam"}')
        parser.add_argument('--learning-rate', default=0.0003, type=float)
        parser.add_argument('--gradient-clip', default=-1, type=float, help='Gradient element clipping value.')
        parser.add_argument('--gradient-norm-clip', default=1, type=float, help='Clips gradient norm to this value.')
        parser.add_argument('--batch-size', default=16, type=int)
        parser.add_argument('--warmup-iterations', default=500, type=int)

        parser.add_argument('-i', '--in-checkpoint', type=str)
        parser.add_argument('-o', '--out-checkpoint', type=str)
        parser.add_argument('-d', '--checkpoint-dir', default='.', type=str)

        parser.add_argument('--mixed-precision', action='store_true')
        return parser

    def __init__(self, args: argparse.Namespace, model, loss, metrics: {}, trn_dataset, val_datasets, tokenizer):
        self.args = args
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.tokenizer = tokenizer

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f'DEVICE {self.device}')

        self.trn_dataset = trn_dataset
        self.val_datasets = val_datasets
        print(args.batch_size)
        self.trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=0, pin_memory=True, drop_last=True)

        self.start_iteration = args.start_iteration
        self.iteration = args.start_iteration
        self.max_iterations = args.max_iterations
        self.view_step = args.view_step
        self.out_checkpoint = args.out_checkpoint
        self.checkpoint_dir = args.checkpoint_dir

        self.gradient_clip = args.gradient_clip
        self.gradient_norm_clip = args.gradient_norm_clip

        checkpoint_path = None
        if args.in_checkpoint is not None:
            checkpoint_path = args.in_checkpoint
        elif args.start_iteration:
            checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_{:06d}.pth".format(args.start_iteration))
        if checkpoint_path is not None:
            logging.info(f"Restore model {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))

        self.model = model.to(self.device)

        optim_config = json.loads(args.optimization_config)
        optim_type = optim_config['type'].lower()
        del optim_config['type']

        params = self.model.parameters()
        #params = self.model.decoder.parameters()
        if optim_type == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=args.learning_rate, **optim_config)
        elif optim_type == 'adamw':
            self.optimizer = torch.optim.AdamW(params, lr=args.learning_rate, **optim_config)
        elif optim_type == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=args.learning_rate, **optim_config)
        else:
            raise ValueError(f'Unknown optimizer type {optim_type}')

        self.tb_writer = SummaryWriter(os.path.join(args.tensorboard_path, args.name))
        self.scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

        self.iteration = args.start_iteration

        self.loss = loss.to(self.device)
        for metric in self.metrics.values():
            metric.to(self.device)

    def change_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def run_training(self, max_iterations: int = None):
        if max_iterations is not None:
            self.max_iterations = max_iterations

        trn_loss_list = []
        train_time_list = []
        start_iter_time = time.time()

        logging.info(f'Start training loop')
        while True:
            for batch in self.trn_loader:
                if self.iteration > self.max_iterations:
                    break

                if self.iteration - self.start_iteration < self.args.warmup_iterations:
                    lr = self.args.learning_rate * (self.iteration - self.start_iteration) / self.args.warmup_iterations
                    self.change_lr(lr)

                # the data shape is (batch_size, 1, seq_len)
                with torch.no_grad():
                    batch_data = batch[0].to(self.device, non_blocking=True).long()
                    batch_labels = batch[1].to(self.device, non_blocking=True).long()
                    batch_masked_labels = batch_labels.clone()
                    if not self.trn_dataset.causal:
                        batch_masked_labels[batch_data != 3] = 0

                self.iteration += 1
                net_t1 = time.time()
                self.optimizer.zero_grad()
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output = self.model(batch_data)[0]
                        trn_loss = self.loss(output, batch_masked_labels)
                        trn_loss = torch.mean(trn_loss)
                    self.scaler.scale(trn_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(batch_data)
                    if isinstance(output, tuple):
                        output, counts, route_prob, n_dropped, route_prob_max = output
                        trn_loss = torch.mean(self.loss(output, batch_masked_labels))
                        load_balancing_loss = self.model.load_balancing_loss(counts, route_prob)
                        print(f'trn_loss: {trn_loss.item()}, load balancing loss: {load_balancing_loss.item()}, dropped: {n_dropped}')
                        trn_loss += 0.01 * load_balancing_loss
                    else:
                        trn_loss = self.loss(output, batch_masked_labels)
                        trn_loss = torch.mean(trn_loss)

                    trn_loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.gradient_clip)
                    if self.gradient_norm_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_norm_clip)
                    self.optimizer.step()

                trn_loss = trn_loss.item()
                trn_loss_list.append(trn_loss)
                # print parameter value statistics for all layers
                if self.iteration % 100 == 0:
                    for name, param in self.model.named_parameters():
                            # print with 4 decimal places and 2 leading spaces to account for the sign
                            print(f'mean: {param.mean().item():6.4f}, std: {param.std().item():6.4f}, max: {param.max().item():6.4f}, min: {param.min().item():6.4f}, {str(param.shape):>10}, {name}')

                #embedding = list(self.model.embed.parameters())[0]
                #grad = embedding.grad
                #embedding = embedding.detach()
                #7last_weights = list(self.model.decoder.parameters())[0].detach()

                #print('VALUES:', ' '.join([f'{x.item():.4f}' for x in embedding[3, :12].cpu().numpy()]))
                #print('GRAD: ', ' '.join([f'{x.item():.4f}' for x in grad[3, :12].cpu().numpy()]))
                #print(f'grad shape: {grad.shape}, mean: {grad.mean().item()}, std: {grad.std().item()}, max: {grad.max().item()}, min: {grad.min().item()}')



                if self.tb_writer:
                    self.tb_writer.add_scalar('Loss/train', trn_loss, self.iteration)

                train_time_list.append(time.time() - net_t1)
                if self.iteration % self.view_step == (self.view_step - 1):
                    step_time = time.time() - start_iter_time
                    if self.out_checkpoint is not None:
                        checkpoint_path = self.out_checkpoint
                    else:
                        if not os.path.exists(self.checkpoint_dir):
                            os.makedirs(self.checkpoint_dir)
                        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_{:06d}.pth".format(self.iteration + 1))
                    torch.save(self.model.state_dict(), checkpoint_path)

                    logging.info(f"ITERATION {self.iteration}")
                    net_speed = (len(trn_loss_list) * batch_data.shape[0] * batch_data.shape[1] ) \
                                / np.sum(train_time_list)
                    fps = len(trn_loss_list) * batch_data.shape[0] / step_time
                    trn_loss_acc = np.mean(trn_loss_list)

                    print(
                        f"TRAIN {self.iteration} loss:{trn_loss_acc:.3f} time:{step_time:.1f} fps::{fps:.1f} net_speed:{int(net_speed)}")

                    for dataset in [self.trn_dataset] + self.val_datasets:
                        self.test_model(dataset)

                    trn_loss_list = []
                    train_time_list = []
                    start_iter_time = time.time()

    def test_model(self, dataset):
        self.model.eval()
        all_loss = []
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
            for i, batch in enumerate(loader):
                batch_data = batch[0].to(self.device, non_blocking=True).long()
                batch_labels = batch[1].to(self.device, non_blocking=True).long()
                batch_masked_labels = batch_labels.clone()
                if not dataset.causal:
                    batch_masked_labels[batch_data != 3] = 0

                output = self.model(batch_data)
                loss = self.loss(output, batch_masked_labels)
                all_loss.append(loss.cpu().item())
                if i == 0:
                    for gt, input, output, i in zip(batch_labels, batch_data, output.cpu().numpy(), range(32)):
                        gt = gt[:128].cpu().numpy()
                        input = input[:128].cpu().numpy()
                        output = output[:128]
                        gt = self.tokenizer.DecodeIds(gt.tolist())
                        print('GT:      ', gt)
                        output = np.argmax(output, axis=1)
                        if not dataset.causal:
                            output[input != 3] = input[input != 3]
                        output = self.tokenizer.DecodeIds(output.tolist())
                        input = self.tokenizer.DecodeIds(input.tolist())
                        print('INPUT:   ', console_transcription_errors(input, gt))
                        print('OUTPUT:  ', console_transcription_errors(output, gt))
                        print('CHANGES: ', console_transcription_errors(output, input))
                        print()

                    if dataset.causal:
                        for gt_tokens in batch_labels[:8]:
                            # sample text from the model base on gt prefix and print it
                            gt_tokens = gt_tokens[:16]
                            prediction_tokens = sample_characters(self.model, gt_tokens[:16], char_count=128, temperature=1.2)
                            prediction = self.tokenizer.DecodeIds(prediction_tokens)
                            gt = self.tokenizer.DecodeIds(gt_tokens.tolist())
                            print('SAMPLING: ' + diff_print.green(gt) + prediction)
                if i > 20:
                    break
        self.model.train()

        loss = np.mean(all_loss)
        if self.tb_writer:
            self.tb_writer.add_scalar(f'Loss_val/{dataset.name}', loss, self.iteration)

        print(f"TEST {dataset.name} {self.iteration} loss:{loss:.3f}")

