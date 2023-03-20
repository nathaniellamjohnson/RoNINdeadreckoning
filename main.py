import json
import os
import sys
import time
from os import path as osp
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')
_nano_to_sec = 1e09
_input_channel, _output_channel = 6, 2
device = 'cpu'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    global device, _output_channel
    import matplotlib.pyplot as plt
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    preds = np.squeeze(network(feat).cpu().detach().numpy())[-vel.shape[0]:, :_output_channel]
    print(preds)


# def test(args, **kwargs):
#     global device, _output_channel
#     import matplotlib.pyplot as plt
#
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#
#     if args.test_path is not None:
#         if args.test_path[-1] == '/':
#             args.test_path = args.test_path[:-1]
#         root_dir = osp.split(args.test_path)[0]
#         test_data_list = [osp.split(args.test_path)[1]]
#     elif args.test_list is not None:
#         root_dir = args.data_dir if args.data_dir else osp.split(args.test_list)[0]
#         with open(args.test_list) as f:
#             test_data_list = [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
#     else:
#         raise ValueError('Either test_path or test_list must be specified.')
#
#     # Load the first sequence to update the input and output size
#     _ = get_dataset(root_dir, [test_data_list[0]], args, mode='test')
#
#     if args.out_dir and not osp.exists(args.out_dir):
#         os.makedirs(args.out_dir)
#
#     with open(osp.join(str(Path(args.model_path).parents[1]), 'config.json'), 'r') as f:
#         model_data = json.load(f)
#
#     if device.type == 'cpu':
#         checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
#     else:
#         checkpoint = torch.load(args.model_path, map_location={model_data['device']: args.device})
#
#     network = get_model(args, **kwargs)
#     network.load_state_dict(checkpoint.get('model_state_dict'))
#     network.eval().to(device)
#     print('Model {} loaded to device {}.'.format(args.model_path, device))
#
#     log_file = None
#     if args.test_list and args.out_dir:
#         log_file = osp.join(args.out_dir, osp.split(args.test_list)[-1].split('.')[0] + '_log.txt')
#         with open(log_file, 'w') as f:
#             f.write(args.model_path + '\n')
#             f.write('Seq traj_len velocity ate rte\n')
#
#     losses_vel = MSEAverageMeter(2, [1], _output_channel)
#     ate_all, rte_all = [], []
#     pred_per_min = 200 * 60
#
#     seq_dataset = get_dataset(root_dir, test_data_list, args, mode='test', **kwargs)
#
#     for idx, data in enumerate(test_data_list):
#         assert data == osp.split(seq_dataset.data_path[idx])[1]
#
#         feat, vel = seq_dataset.get_test_seq(idx)
#         feat = torch.Tensor(feat).to(device)
#         # IMPORTANT LINE HERE
#         preds = np.squeeze(network(feat).cpu().detach().numpy())[-vel.shape[0]:, :_output_channel]
#
#         ind = np.arange(vel.shape[0])
#         vel_losses = np.mean((vel - preds) ** 2, axis=0)
#         losses_vel.add(vel, preds)
#
#         print('Reconstructing trajectory')
#         pos_pred, gv_pred, _ = recon_traj_with_preds_global(seq_dataset, preds, ind=ind, type='pred', seq_id=idx)
#         pos_gt, gv_gt, _ = recon_traj_with_preds_global(seq_dataset, vel, ind=ind, type='gt', seq_id=idx)
#
#         if args.out_dir is not None and osp.isdir(args.out_dir):
#             np.save(osp.join(args.out_dir, '{}_{}.npy'.format(data, args.type)),
#                     np.concatenate([pos_pred, pos_gt], axis=1))
#
#         ate = compute_absolute_trajectory_error(pos_pred, pos_gt)
#         if pos_pred.shape[0] < pred_per_min:
#             ratio = pred_per_min / pos_pred.shape[0]
#             rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pos_pred.shape[0] - 1) * ratio
#         else:
#             rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pred_per_min)
#         pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)
#         ate_all.append(ate)
#         rte_all.append(rte)
#
#         print('Sequence {}, Velocity loss {} / {}, ATE: {}, RTE:{}'.format(data, vel_losses, np.mean(vel_losses), ate,
#                                                                            rte))
#         log_line = format_string(data, np.mean(vel_losses), ate, rte)
#
#         if not args.fast_test:
#             kp = preds.shape[1]
#             if kp == 2:
#                 targ_names = ['vx', 'vy']
#             elif kp == 3:
#                 targ_names = ['vx', 'vy', 'vz']
#
#             plt.figure('{}'.format(data), figsize=(16, 9))
#             plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
#             plt.plot(pos_pred[:, 0], pos_pred[:, 1])
#             plt.plot(pos_gt[:, 0], pos_gt[:, 1])
#             plt.title(data)
#             plt.axis('equal')
#             plt.legend(['Predicted', 'Ground truth'])
#             plt.subplot2grid((kp, 2), (kp - 1, 0))
#             plt.plot(pos_cum_error)
#             plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
#             for i in range(kp):
#                 plt.subplot2grid((kp, 2), (i, 1))
#                 plt.plot(ind, preds[:, i])
#                 plt.plot(ind, vel[:, i])
#                 plt.legend(['Predicted', 'Ground truth'])
#                 plt.title('{}, error: {:.6f}'.format(targ_names[i], vel_losses[i]))
#             plt.tight_layout()
#
#             if args.show_plot:
#                 plt.show()
#
#             if args.out_dir is not None and osp.isdir(args.out_dir):
#                 plt.savefig(osp.join(args.out_dir, '{}_{}.png'.format(data, args.type)))
#
#         if log_file is not None:
#             with open(log_file, 'a') as f:
#                 log_line += '\n'
#                 f.write(log_line)
#
#         plt.close('all')
#
#     ate_all = np.array(ate_all)
#     rte_all = np.array(rte_all)
#
#     measure = format_string('ATE', 'RTE', sep='\t')
#     values = format_string(np.mean(ate_all), np.mean(rte_all), sep='\t')
#     print(measure, '\n', values)
#
#     if log_file is not None:
#         with open(log_file, 'a') as f:
#             f.write(measure + '\n')
#             f.write(values)
