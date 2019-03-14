# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from NME import NME

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def copy_prev_models(prev_models_dir, model_dir):
    import shutil

    vc_folder = '/hdfs/' \
                + '/' + os.environ['PHILLY_VC']
    source = prev_models_dir
    # If path is set as "sys/jobs/application_1533861538020_2366/models" prefix with the location of vc folder
    source = vc_folder + '/' + source if not source.startswith(vc_folder) \
        else source
    destination = model_dir

    if os.path.exists(source) and os.path.exists(destination):
        for file in os.listdir(source):
            source_file = os.path.join(source, file)
            destination_file = os.path.join(destination, file)
            if not os.path.exists(destination_file):
                print("=> copying {0} to {1}".format(
                    source_file, destination_file))
            shutil.copytree(source_file, destination_file)
    else:
        print('=> {} or {} does not exist'.format(source, destination))


def main():
    args = parse_args()
    update_config(cfg, args)

    if args.prevModelDir and args.modelDir:
        # copy pre models for philly
        copy_prev_models(args.prevModelDir, args.modelDir)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    writer_dict['writer'].add_graph(model, (dump_input,))

    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    # if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
    #     logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    #     checkpoint = torch.load(checkpoint_file)
    #     begin_epoch = checkpoint['epoch']
    #     best_perf = checkpoint['perf']
    #     last_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     logger.info("=> loaded checkpoint '{}' (epoch {})".format(
    #         checkpoint_file, checkpoint['epoch']))


    # checkpoint = torch.load('output/jd/pose_hrnet/crop_face/checkpoint.pth')
    # model.load_state_dict(checkpoint['state_dict'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        # evaluate on validation set
        # perf_indicator = validate(
        #     cfg, valid_loader, valid_dataset, model, criterion,
        #     final_output_dir, tb_log_dir, writer_dict
        # )
        #
        # if perf_indicator >= best_perf:
        #     best_perf = perf_indicator
        #     best_model = True
        # else:
        #     best_model = False

        # import tqdm
        # import cv2
        # import numpy as np
        # from lib.utils.imutils import im_to_numpy, im_to_torch
        # flip = True
        # full_result = []
        # for i, (inputs,target, target_weight, meta) in enumerate(valid_loader):
        #     with torch.no_grad():
        #         input_var = torch.autograd.Variable(inputs.cuda())
        #         if flip == True:
        #             flip_inputs = inputs.clone()
        #             for i, finp in enumerate(flip_inputs):
        #                 finp = im_to_numpy(finp)
        #                 finp = cv2.flip(finp, 1)
        #                 flip_inputs[i] = im_to_torch(finp)
        #             flip_input_var = torch.autograd.Variable(flip_inputs.cuda())
        #
        #         # compute output
        #         refine_output = model(input_var)
        #         score_map = refine_output.data.cpu()
        #         score_map = score_map.numpy()
        #
        #         if flip == True:
        #             flip_output = model(flip_input_var)
        #             flip_score_map = flip_output.data.cpu()
        #             flip_score_map = flip_score_map.numpy()
        #
        #             for i, fscore in enumerate(flip_score_map):
        #                 fscore = fscore.transpose((1, 2, 0))
        #                 fscore = cv2.flip(fscore, 1)
        #                 fscore = list(fscore.transpose((2, 0, 1)))
        #                 for (q, w) in train_dataset.flip_pairs:
        #                     fscore[q], fscore[w] = fscore[w], fscore[q]
        #                 fscore = np.array(fscore)
        #                 score_map[i] += fscore
        #                 score_map[i] /= 2
        #
        #         # ids = meta['imgID'].numpy()
        #         # det_scores = meta['det_scores']
        #         for b in range(inputs.size(0)):
        #             # details = meta['augmentation_details']
        #             # imgid = meta['imgid'][b]
        #             # print(imgid)
        #             # category = meta['category'][b]
        #             # print(category)
        #             single_result_dict = {}
        #             single_result = []
        #
        #             single_map = score_map[b]
        #             r0 = single_map.copy()
        #             r0 /= 255
        #             r0 += 0.5
        #             v_score = np.zeros(106)
        #             for p in range(106):
        #                 single_map[p] /= np.amax(single_map[p])
        #                 border = 10
        #                 dr = np.zeros((112 + 2 * border, 112 + 2 * border))
        #                 dr[border:-border, border:-border] = single_map[p].copy()
        #                 dr = cv2.GaussianBlur(dr, (7, 7), 0)
        #                 lb = dr.argmax()
        #                 y, x = np.unravel_index(lb, dr.shape)
        #                 dr[y, x] = 0
        #                 lb = dr.argmax()
        #                 py, px = np.unravel_index(lb, dr.shape)
        #                 y -= border
        #                 x -= border
        #                 py -= border + y
        #                 px -= border + x
        #                 ln = (px ** 2 + py ** 2) ** 0.5
        #                 delta = 0.25
        #                 if ln > 1e-3:
        #                     x += delta * px / ln
        #                     y += delta * py / ln
        #                 x = max(0, min(x, 112 - 1))
        #                 y = max(0, min(y, 112 - 1))
        #                 resy = float((4 * y + 2) / 112 * (450))
        #                 resx = float((4 * x + 2) / 112 * (450))
        #                 # resy = float((4 * y + 2) / cfg.data_shape[0] * (450))
        #                 # resx = float((4 * x + 2) / cfg.data_shape[1] * (450))
        #                 v_score[p] = float(r0[p, int(round(y) + 1e-10), int(round(x) + 1e-10)])
        #                 single_result.append(resx)
        #                 single_result.append(resy)
        #             if len(single_result) != 0:
        #                 result = []
        #                 # result.append(imgid)
        #                 j = 0
        #                 while j < len(single_result):
        #                     result.append(float(single_result[j]))
        #                     result.append(float(single_result[j + 1]))
        #                     j += 2
        #                 full_result.append(result)
        model.eval()

        import numpy as np
        from core.inference import get_final_preds
        from utils.transforms import flip_back
        import csv

        num_samples = len(valid_dataset)
        all_preds = np.zeros(
            (num_samples, 106, 3),
            dtype=np.float32
        )
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        imgnums = []
        idx = 0
        full_result = []
        with torch.no_grad():
            for i, (input, target, target_weight, meta) in enumerate(valid_loader):
                # compute output
                outputs = model(input)
                if isinstance(outputs, list):
                    output = outputs[-1]
                else:
                    output = outputs

                if cfg.TEST.FLIP_TEST:
                    # this part is ugly, because pytorch has not supported negative index
                    # input_flipped = model(input[:, :, :, ::-1])
                    input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                    input_flipped = torch.from_numpy(input_flipped).cuda()
                    outputs_flipped = model(input_flipped)

                    if isinstance(outputs_flipped, list):
                        output_flipped = outputs_flipped[-1]
                    else:
                        output_flipped = outputs_flipped

                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                               valid_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if cfg.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                loss = criterion(output, target, target_weight)

                num_images = input.size(0)
                # measure accuracy and record loss


                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                # print(c.shape)
                # print(s.shape)
                # print(c[:3, :])
                # print(s[:3, :])
                score = meta['score'].numpy()

                preds, maxvals = get_final_preds(
                    cfg, output.clone().cpu().numpy(), c, s)

                # print(preds.shape)
                for b in range(input.size(0)):
                    result = []
                    # pic_name=meta['image'][b].split('/')[-1]
                    # result.append(pic_name)
                    for points in range(106):
                        # result.append(str(int(preds[b][points][0])) + ' ' + str(int(preds[b][points][1])))
                        result.append(float(preds[b][points][0]))
                        result.append(float(preds[b][points][1]))

                    full_result.append(result)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])

                idx += num_images




        # with open('res.csv', 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(full_result)
        gt = []
        with open("/home/sk49/workspace/cy/jd/val.txt") as f:
            for line in f.readlines():
                rows = list(map(float, line.strip().split(' ')[1:]))
                gt.append(rows)

        error = 0
        for i in range(len(gt)):
            error = NME(full_result[i], gt[i]) + error
        print(error)

        log_file = []
        log_file.append([epoch, optimizer.state_dict()['param_groups'][0]['lr'], error])

        with open('log_file.csv', 'a', newline='') as f:
            writer1 = csv.writer(f)
            writer1.writerows(log_file)
            # logger.close()

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
