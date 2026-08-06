import time
import random
import csv
import os
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
# from util import util
import numpy as np
import torch
from tqdm import tqdm

def print_current_losses(epoch, i, losses, t, t_data):
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)
    print(message)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    history_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss_history.csv')

    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_loss_sums = {}
        epoch_batches = 0

        pbar = tqdm(enumerate(dataset), total=len(data_loader.dataloader), desc=f'Epoch {epoch}', ncols=120, unit='batch')

        for i, data in pbar:
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()
            losses = model.get_current_losses()
            for key, value in losses.items():
                epoch_loss_sums[key] = epoch_loss_sums.get(key, 0.0) + float(value)
            epoch_batches += 1

            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.batch_size
                print_current_losses(epoch, epoch_iter, losses, t, t_data)
                pbar.set_postfix({k: f'{v:.3f}' for k, v in losses.items()})

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        # done from epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('epoch_%d' % epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        history_exists = os.path.isfile(history_path) and os.path.getsize(history_path) > 0
        history_row = {
            'epoch': epoch,
            'batches': epoch_batches,
            'lr': model.optimizers[0].param_groups[0]['lr'],
        }
        history_row.update({
            key: value / max(1, epoch_batches)
            for key, value in epoch_loss_sums.items()
        })
        with open(history_path, 'a', newline='', encoding='utf-8') as history_file:
            writer = csv.DictWriter(history_file, fieldnames=list(history_row.keys()))
            if not history_exists:
                writer.writeheader()
            writer.writerow(history_row)
        print('loss history saved to %s' % history_path)
        model.update_learning_rate()
