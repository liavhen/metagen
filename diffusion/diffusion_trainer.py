"""Train models on a given dataset."""
import os
import numpy as np
import torch.nn
import wandb
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import utils
from utils.paths import *
import pandas as pd
import copy, shutil
from diffusion.sample import sample, compute_actual_scatterings, compute_metrics, log_results, plot_results
from math import gcd
from functools import reduce

class DiffusionTrainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, sdf, datasets, args, logger):
        # Attributes
        self.model = model
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.sdf = sdf
        self.args = args
        self.logger = logger

        # Data
        self.batch_size = args.batch_size
        self.train_set, self.test_set = datasets

        self.train_loader = DataLoader(self.train_set, self.args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True, prefetch_factor=4)
        self.test_loader = DataLoader(self.test_set, self.args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True, prefetch_factor=4)

        self.train_iter = None
        self.test_iter = None

        # Performance Monitoring
        self.curr_step = 0
        self.best_step = 0
        self.running_train_loss = []
        self.running_eval_loss = []
        self.loss_per_sigma = dict(loss=[], sigma=[], time_step=[])
        self.sampled_steps = []
        self.diffraction_errors = dict(
            # Relative Errors
            test_t_mean_relative_error=[], target_t_mean_relative_error=[],        # T mean
            test_r_mean_relative_error=[], target_r_mean_relative_error=[],        # R mean
            test_te_mean_relative_error=[], target_te_mean_relative_error=[],      # Te mean
            test_tm_mean_relative_error=[], target_tm_mean_relative_error=[],      # Tm mean
            # train_relative_t_std=[], test_relative_t_std=[], target_relative_t_std=[],
            # train_relative_r_mean=[], test_relative_r_mean=[], target_relative_r_mean=[],
            # train_relative_r_std=[], test_relative_r_std=[], target_relative_r_std=[],

            # # NRMS Errors
            # train_nrms_t_mean=[], test_nrms_t_mean=[], target_nrms_t_mean=[],
            # train_nrms_t_std=[], test_nrms_t_std=[], target_nrms_t_std=[],
            # train_nrms_r_mean=[], test_nrms_r_mean=[], target_nrms_r_mean=[],
            # train_nrms_r_std=[], test_nrms_r_std=[], target_nrms_r_std=[],

            # Uniformity Errors,
            test_t_mean_ue_error=[], target_t_mean_ue_error=[],        # T mean
            test_r_mean_ue_error=[], target_mean_ue_error=[],        # R mean
            test_te_mean_ue_error=[], target_te_mean_ue_error=[],      # Te mean
            test_tm_mean_ue_error=[], target_tm_mean_ue_error=[],      # Tm mean

            # test_ue_t_mean=[], target_ue_t_mean=[],
            # test_ue_r_mean=[], target_ue_r_mean=[],
        )

        self.monitored_blocks = None
        self.grad_sampled_steps = None
        self.grads_l1_so_far = None
        self.grads_per_w_so_far = None
        self.init_grad_monitor()

        # Saving
        self.savepath = args.outdir
        self.best_savepath = os.path.join(self.savepath, 'best_model')
        self.last_savepath = os.path.join(self.savepath, 'last_model')
        os.makedirs(self.savepath, exist_ok=True)
        os.makedirs(self.best_savepath, exist_ok=True)
        os.makedirs(self.last_savepath, exist_ok=True)

    def init_grad_monitor(self):
        self.model.train()
        self.monitored_blocks = [n for n, p in self.model.named_parameters() if p.requires_grad and "bias" not in n]
        self.grad_sampled_steps = []
        self.grads_l1_so_far = [[] for _ in self.monitored_blocks]
        self.grads_per_w_so_far = [[] for _ in self.monitored_blocks]

    def do_forward(self, sample):
        layer, scattering = sample['layer'].cuda(non_blocking=True), sample['scattering'].cuda(non_blocking=True)
        return self.loss_fn(net=self.model, images=layer, labels=scattering, augment_pipe=self.sdf)

    def train_step(self):
        self.model.train()
        self.optimizer.train()
        step_loss = torch.tensor(0.0, device=self.args.device)
        for _ in range(self.args.accumulation_steps):
            try:
                sample = next(self.train_iter)
            except:
                self.train_iter = iter(self.train_loader)
                sample = next(self.train_iter)
            loss, _ = self.do_forward(sample)
            loss /= self.args.accumulation_steps
            loss.backward()
            step_loss += loss
        return step_loss

    @torch.no_grad()
    def eval_step(self):
        self.model.eval()
        step_loss = torch.tensor(0.0, device=self.args.device)
        for _ in range(self.args.accumulation_steps):
            try:
                sample = next(self.test_iter)
            except:
                self.test_iter = iter(self.test_loader)
                sample = next(self.test_iter)
            loss, _ = self.do_forward(sample)
            step_loss += loss / self.args.accumulation_steps
        return step_loss

    def train(self):
        min_error = np.inf

        # Check if all the specified intervals have a common divisor  
        logging_intervals = [self.args.log_every, self.args.sample_every, self.args.save_every]
        common_divisor = reduce(gcd, logging_intervals)
        assert common_divisor != 1 or min(logging_intervals) == 1, "All logging frequencies (log_every, sample_every, save_every) must have a common divisor other than 1 (unless one of the is 1)"

        tick = utils.now()
        for _ in range(self.curr_step, self.args.num_steps):

            self.optimizer.zero_grad(set_to_none=True)
            train_loss = self.train_step()

            # Log periodically and save last.
            if self.curr_step % self.args.log_every == 0:
                eval_loss = self.eval_step()
                self.save_checkpoints(path=self.last_savepath)
                self.save_results(path=self.last_savepath)
                log_data = dict(step=self.curr_step, train_loss=train_loss.detach().item(), eval_loss=eval_loss.detach().item())
                self.log(log_data)

                # Sample model and save if best.
                if self.curr_step % self.args.sample_every == 0 and self.curr_step != 0:
                    
                    # Save periodic snapshots.
                    if self.curr_step % self.args.save_every == 0 and self.curr_step != 0:
                        self.savepath = os.path.join(self.args.outdir, f'ckpt_{self.curr_step}')
                        os.makedirs(self.savepath, exist_ok=True)
                        self.save_checkpoints(path=self.savepath)
                        self.save_results(path=self.savepath)
                    else:
                        self.savepath = self.last_savepath

                    self.sample_model(model_name='ema', savepath=self.savepath)
                    self.sampled_steps.append(self.curr_step)

                    log_data['test_t_mean_relative_error'] = self.diffraction_errors['test_t_mean_relative_error'][-1]
                    log_data['target_t_mean_relative_error'] = self.diffraction_errors['target_t_mean_relative_error'][-1]
                    log_data['target_t_mean_ue_error'] = self.diffraction_errors['target_t_mean_ue_error'][-1]

                    last_error = self.diffraction_errors['target_t_mean_relative_error'][-1]

                    if self.args.save_best and last_error < min_error:
                        min_error = last_error
                        shutil.copytree(self.savepath, self.best_savepath, dirs_exist_ok=True)
                        # self.save_checkpoints(path=self.best_savepath)
                        # self.save_results(path=self.best_savepath)
                        # self.save_results(path=self.best_savepath)
                
                log_data[f'time_per_{self.args.log_every}_steps'] = str(utils.now() - tick)
                self.logger.info(utils.print_dict_beautifully(log_data))
                tick = utils.now()
                

            # # Log periodically and save last.
            # if self.curr_step % self.args.log_every == 0:
            #     eval_loss = self.eval_step()
            #     log_data = dict(step=self.curr_step, train_loss=train_loss.detach().item(), eval_loss=eval_loss.detach().item())
            #     self.log(log_data)

                # if self.curr_step % self.args.sample_every == 0 and self.curr_step != 0:
                #     log_data['test_t_mean_relative_error'] = self.diffraction_errors['test_t_mean_relative_error'][-1]
                #     log_data['target_t_mean_relative_error'] = self.diffraction_errors['target_t_mean_relative_error'][-1]
                #     log_data['target_t_mean_ue_error'] = self.diffraction_errors['target_t_mean_ue_error'][-1]

                    # log_data[f'time_per_{self.args.log_every}_steps'] = str(utils.now() - tick)
                    # tick = utils.now()
                    # self.logger.info(utils.print_dict_beautifully(log_data))
                    # self.save_checkpoints(path=self.last_savepath)
                    # self.save_results(path=self.last_savepath)

                # # Log gradients
                # if self.curr_step != 0:
                #     grads_l1 = {f'Gradients L1 Norm - {n}': (torch.norm(p.grad, p=1)).item() for n, p in self.model.named_parameters() if p.requires_grad and "bias" not in n}
                #     grads_per_w = {f'Gradients Normalized - {n}': (torch.norm(p.grad, p=1)/(torch.norm(p, p=1) + 1e-13)).item() for n, p in self.model.named_parameters() if p.requires_grad and "bias" not in n}
                #     self.log(grads_l1)
                #     self.log(grads_per_w)


            # Update Weights
            for g in self.optimizer.param_groups:
                g['lr'] = self.args.lr * min(self.curr_step / self.args.warmup, 1)
            for param in self.model.module.denoiser.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Update EMA
            ema_halflife = self.args.ema_halflife
            if self.args.ema_warmup_ratio is not None:
                ema_halflife = min(self.args.ema_halflife, self.curr_step * self.args.ema_warmup_ratio)
            ema_beta = 0.5 ** (self.batch_size / max(ema_halflife, 1e-8))
            for p_ema, p_model in zip(self.ema.parameters(), self.model.parameters()):
                p_ema.copy_(p_model.detach().lerp(p_ema, ema_beta))

            self.curr_step += 1

    def sample_model(self, model_name, savepath):
        assert model_name in ["ema", "model"]
        if model_name == "ema":
            model = self.ema
        else:
            model = self.model

        eval_batch_size = 100

        for dataloader_type in ['test', 'target']:

            dataloader = self.train_loader if dataloader_type == 'train' else self.test_loader if dataloader_type == 'test' else None
            ood = dataloader_type == 'target'
            ood_repeat = 1

            results = sample(
                data_cfg        = self.args.data_cfg,
                model           = model.module if isinstance(model, torch.nn.DataParallel) else model,
                model_kwargs    = {'model_type': self.args.model_type, },
                dataloader      = dataloader, 
                data_kwargs     = {'data_type': 'ood' if ood else 'test', 'eval_batch_size': eval_batch_size if not ood else 1},
                repeat          = (ood_repeat if ood else 1),
            )

            results = compute_actual_scatterings(results, self.args.data_cfg, self.logger, n_jobs=1, verbose=False)
            metrics = compute_metrics(results, self.args.data_cfg)

            self.diffraction_errors[f'{dataloader_type}_t_mean_relative_error'].append(metrics['T_mean_relative_error'])
            # self.diffraction_errors[f'{dataloader_type}_relative_t_std'].append(metrics['relative_t_std'])
            self.diffraction_errors[f'{dataloader_type}_r_mean_relative_error'].append(metrics['R_mean_relative_error'])
            # self.diffraction_errors[f'{dataloader_type}_relative_r_std'].append(metrics['relative_r_std'])

            # self.diffraction_errors[f'{dataloader_type}_relative_t_mean'].append(metrics['relative_t_mean'])
            # self.diffraction_errors[f'{dataloader_type}_relative_t_std'].append(metrics['relative_t_std'])
            # self.diffraction_errors[f'{dataloader_type}_relative_r_mean'].append(metrics['relative_r_mean'])
            # self.diffraction_errors[f'{dataloader_type}_relative_r_std'].append(metrics['relative_r_std'])

            # self.diffraction_errors[f'{dataloader_type}_nrms_t_mean'].append(np.mean(metrics['nrms_t_mean']))
            # self.diffraction_errors[f'{dataloader_type}_nrms_t_std'].append(np.std(metrics['nrms_t_std']))
            # self.diffraction_errors[f'{dataloader_type}_nrms_r_mean'].append(np.mean(metrics['nrms_r_mean']))
            # self.diffraction_errors[f'{dataloader_type}_nrms_r_std'].append(np.std(metrics['nrms_r_std']))

            self.diffraction_errors[f'{dataloader_type}_t_mean_ue_error'].append(np.mean(metrics['T_mean_ue_error']))
            # self.diffraction_errors[f'{dataloader_type}_ue_r_mean'].append(np.mean(metrics['ue_r_mean']))


            errors_log = {
                f'{dataloader_type}_relative_Tte_error' : metrics['Tte_relative_errors'].mean().item(),
                f'{dataloader_type}_relative_Rte_error' : metrics['Rte_relative_errors'].mean().item(),
                f'{dataloader_type}_relative_Ttm_error' : metrics['Ttm_relative_errors'].mean().item(),
                f'{dataloader_type}_relative_Rtm_error' : metrics['Rtm_relative_errors'].mean().item(),
                f'{dataloader_type}_relative_T_error'   : metrics['T_mean_relative_error'],
                f'{dataloader_type}_relative_R_error'   : metrics['R_mean_relative_error'],
                f'{dataloader_type}_relative_TE_error'  : metrics['TE_mean_relative_error'],
                f'{dataloader_type}_relative_TM_error'  : metrics['TM_mean_relative_error'],

                f'{dataloader_type}_UE_Tte_error': metrics['Tte_ue_errors'].mean().item(),
                f'{dataloader_type}_UE_Rte_error': metrics['Rte_ue_errors'].mean().item(),
                f'{dataloader_type}_UE_Ttm_error': metrics['Ttm_ue_errors'].mean().item(),
                f'{dataloader_type}_UE_Rtm_error': metrics['Rtm_ue_errors'].mean().item(),
                f'{dataloader_type}_UE_T_error':   metrics['T_mean_ue_error'],

                
                # f'{dataset_type}_relative_r_error': metrics['relative_r_mean'],
                # f'{dataloader_type}_nrms_t_error':     metrics['nrms_t_mean'],
                # f'{dataset_type}_nrms_r_error':     metrics['nrms_r_mean'],
                # f'{dataloader_type}_ue_t_mean':        metrics['ue_t_mean'],
                # f'{dataset_type}_ue_r_mean':        metrics['ue_r_mean'],

            }

            self.log(errors_log)

            # Save images
            imgs_savepath = os.path.join(savepath, f'sampling_{model_name}_{dataloader_type}')
            os.makedirs(imgs_savepath, exist_ok=True)
            log_results(results, imgs_savepath)
            if dataloader_type == 'target':
                # Save image with best results
                lst4plot = []
                t_errors, t_indices = metrics['Tte_relative_errors'].reshape(ood_repeat, -1).min(dim=0)
                for b in range(len(t_errors)):
                    best_attempt = t_indices[b]
                    best_layer = utils.viewable(results[str(best_attempt.item())]['layer'][b:b + 1])
                    best_sample = utils.viewable(results[str(best_attempt.item())]['sample'][-1][b:b + 1], sampled_from_model=True)
                    best_desired_s = results[str(best_attempt.item())]['desired_scatterings'][b:b+1]
                    best_actual_s = results[str(best_attempt.item())]['actual_scatterings'][b:b+1]
                    lam = results[str(best_attempt.item())]['scattering'][:, -1].cpu().numpy()[b]
                    lst4plot.append((t_errors[b], best_layer, best_sample, best_desired_s, best_actual_s, lam))
                metric_name = 'TE_relative'
                plot_results(lst4plot, self.args.data_cfg, metric_name, imgs_savepath, text_labels=False)

    def resume_training_state(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.logger.info(f"\nResuming training from {path} (after training {checkpoint['step']} timesteps)...")
        self.curr_step = checkpoint['step']
        self.running_train_loss = checkpoint['running_train_loss']
        self.running_eval_loss = checkpoint['running_eval_loss']
        self.diffraction_errors = checkpoint['diffraction_errors']
        self.sampled_steps = checkpoint['sampled_steps']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.model.load_state_dict(checkpoint['model'])
        self.ema.load_state_dict(checkpoint['ema'])

    def load_pretrained_model(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.logger.info(f"\nLoading pretrained weights from {path} to start training...")
        self.model.load_state_dict(checkpoint['ema'])
        self.ema.load_state_dict(checkpoint['ema'])

    def save_checkpoints(self, path):
        os.makedirs(os.path.join(path, 'ckpts'), exist_ok=True)
        state = {
            'running_train_loss': self.running_train_loss,
            'running_eval_loss': self.running_eval_loss,
            'step': self.curr_step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'sampled_steps': self.sampled_steps,
            'diffraction_errors': self.diffraction_errors
        }
        ckpts_path = os.path.join(path, 'ckpts/', f'checkpoint.pth')
        os.makedirs(path, exist_ok=True)
        torch.save(state, ckpts_path)

    def save_results(self, path):
        train_sample = next(iter(self.train_loader))
        test_sample = next(iter(self.test_loader))
        self.model.eval()

        os.makedirs(os.path.join(path, 'train'), exist_ok=True)
        os.system(f"rm -rf {os.path.join(path, 'train')}/*")   # remove current content
        _, results = self.do_forward(train_sample)
        utils.save_images_batch(batch=results, step=self.curr_step,
                                path=os.path.join(path, 'train'))

        os.makedirs(os.path.join(path, 'test'), exist_ok=True)
        os.system(f"rm -rf {os.path.join(path, 'test')}/*")  # remove current content
        _, results = self.do_forward(test_sample)
        utils.save_images_batch(batch=results, step=self.curr_step,
                                path=os.path.join(path, 'test'))

    def save_loss_curve(self):
        pd.DataFrame.from_dict(self.loss_per_sigma).to_csv(os.path.join(self.savepath, f'loss_per_sigma.csv'))

        # Linear interpolation of the evaluation running loss
        if len(self.running_eval_loss) > 1:
            tmp = np.linspace(self.running_eval_loss[-2], self.running_eval_loss[-1], self.args.log_every + 1).tolist()
            self.running_eval_loss = self.running_eval_loss[:-1] + tmp[1:]
            del tmp

        f1 = plt.figure()
        plt.plot(self.running_train_loss, label='Train Loss')
        plt.plot(self.running_eval_loss, label='Eval Loss')
        plt.legend()
        plt.title(f'{self.args.name} | Loss Curve After {self.curr_step} Steps')
        plt.savefig(os.path.join(self.savepath, 'loss_curve.png'))
        plt.close(f1)
        if len(self.running_eval_loss) > 300 and len(self.running_train_loss) > 300:
            f2 = plt.figure()
            plt.plot(np.arange(self.curr_step - 300, self.curr_step), self.running_train_loss[-300:], label='Train Loss')
            plt.plot(np.arange(self.curr_step - 300, self.curr_step), self.running_eval_loss[-300:], label='Eval Loss')
            plt.legend()
            plt.title(f'{self.args.name} | Recent Loss Curve After {self.curr_step} Steps')
            plt.savefig(os.path.join(self.savepath, 'recent_loss_curve.png'))
            plt.close(f2)
        plt.cla()

    def save_error_curves(self):
        x = np.array(self.sampled_steps)
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        for i, d in enumerate(['t', 'r']):
            if d == 'r' and self.args.data_cfg.use_t_only:
                continue
            elif d == 't' and self.args.data_cfg.use_r_only:
                continue
            y1 = np.array(self.diffraction_errors[f'train_relative_{d}_mean'])
            e1 = np.array(self.diffraction_errors[f'train_relative_{d}_std'])
            y2 = np.array(self.diffraction_errors[f'test_relative_{d}_mean'])
            e2 = np.array(self.diffraction_errors[f'test_relative_{d}_std'])
            # ax[i].plot(x, y1, label=f'Train', color='#1B2ACC')
            # ax[i].fill_between(x, y1 - e1, y1 + e1, alpha=0.3, facecolor='#089FFF')
            ax[i].plot(x, y2, label=f'Test', color='#CC4F1B')
            ax[i].fill_between(x, y2 - e2, y2 + e2, alpha=0.3, facecolor='#FF9848')
            ax[i].set_title(f'Diffraction Errors - {d.capitalize()} ')
            ax[i].legend()
            ax[i].set_xlabel('Time Step')
            ax[i].set_ylabel('Mean Relative Error of Diffraction')
        plt.savefig(os.path.join(self.savepath, 'diffraction_errors.png'))
        plt.close(f)

    def log(self, data):
        assert isinstance(data, dict), "data must be a dictionary!"
        if self.args.log:
            wandb.log(data, step=self.curr_step)
