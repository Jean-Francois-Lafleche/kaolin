import argparse
import json
import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from kaolin.datasets import shapenet, modelnet
from kaolin.conversions.voxelgridconversions import extract_odms
from utils import up_sample, upsample_omd, to_occupancy_map
import kaolin as kal
from kaolin.models.VoxelSuperresODM import SuperresNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, help='Path to data directory.')
parser.add_argument('--cache-dir', type=str, default='cache', help='Path to data directory.')
parser.add_argument('--data-source', type=str, choices=['shapenet', 'modelnet'], help='Data source to use.')
parser.add_argument('--expid', type=str, default='MVD', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--epochs', type=int, default=30, help='Number of train epochs.')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--print-every', type=int, default=20, help='Print frequency (batches).')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--resume', choices=['best', 'recent'], default=None,
                    help='Choose which weights to resume training from (None to start from random initialization.)')
args = parser.parse_args()


# Data
if args.data_source == 'shapenet':
    train_set = shapenet.ShapeNet_Voxels(root=args.data_root, cache_dir=args.cache_dir,
                                         categories=args.categories, train=True, split=.97,
                                         resolutions=[32, 128])
    valid_set = shapenet.ShapeNet_Voxels(root=args.data_root, cache_dir=args.cache_dir,
                                         categories=args.categories, train=False, split=.97,
                                         resolutions=[32, 128])
elif args.data_source == 'modelnet':
    train_set = modelnet.ModelNetVoxels(root=args.data_root, cache_dir=args.cache_dir,
                                        categories=args.categories, train=True, split=.97,
                                        resolutions=[32, 128], device=args.device)

    valid_set = modelnet.ModelNetVoxels(root=args.data_root, cache_dir=args.cache_dir,
                                        categories=args.categories, train=False, split=.97,
                                        resolutions=[32, 128], device=args.device)

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=8)
dataloader_val = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=8)


# Create log directory, if it doesn't already exist
logdir = os.path.join(args.logdir, args.expid)
if not os.path.isdir(logdir, 'occ'):
    os.makedirs(logdir)
    print('Created dir:', logdir)
if not os.path.isdir(logdir, 'residual'):
    os.makedirs(logdir)
    print('Created dir:', logdir)

# Log all commandline args
with open(os.path.join(logdir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


class Engine(object):
    """Engine that runs training and inference.
    Args
        model_name (str): Name of model being trained. ['occ', 'residual']
        cur_epoch (int): Current epoch.
        print_every (int): How frequently (# batches) to print loss.
        validate_every (int): How frequently (# epochs) to run validation.
    """

    def __init__(self, model_name, print_every=1, resume_name=None):
        assert model_name in ['occ', 'residual']
        self.cur_epoch = 0
        self.train_loss = []
        self.val_score = []
        self.bestval = 0
        self.model_name = model_name

        if resume_name:
            self.load(resume_name)

    def train(self):
        loss_epoch = 0.
        num_batches = 0

        # Train loop
        for i, sample in enumerate(tqdm(dataloader_train), 0):
            data = sample['data']

            optimizer.zero_grad()
            pred_odms = self._get_pred(data)

            loss = self._get_loss(pred_odms, data)
            loss.backward()
            loss_epoch += float(loss.item())

            # logging
            num_batches += 1
            if i % args.print_every == 0:
                tqdm.write(f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}: Loss: {float(loss.item())}')

            optimizer.step()

        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self):
        model.eval()
        with torch.no_grad():	
            iou_epoch = 0.
            iou_NN_epoch = 0.
            num_batches = 0
            loss_epoch = 0.

            # Validation loop
            for i, sample in enumerate(tqdm(dataloader_val), 0):
                data = sample['data']
                pred_odms = self._get_pred(data)

                loss = self._get_loss(pred_odms, data)

                loss_epoch += float(loss.item())

                iou = self._calculate_iou(pred_odms, data)
                iou_epoch += iou

                # logging
                num_batches += 1
                if i % args.print_every == 0:
                    out_iou = iou_epoch.item() / float(num_batches)
                    out_iou_NN = iou_NN_epoch.item() / float(num_batches)
                    tqdm.write(f'[VAL] Epoch {self.cur_epoch:03d}, Batch {i:03d}: IoU: {out_iou}, Iou Base: {out_iou_NN}')

            out_iou = iou_epoch.item() / float(num_batches)
            out_iou_NN = iou_NN_epoch.item() / float(num_batches)
            tqdm.write(f'[VAL Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: IoU: {out_iou}, Iou Base: {out_iou_NN}')

            loss_epoch = loss_epoch / num_batches
            self.val_score.append(out_iou)

    def load(self, resume_name):
        model.load_state_dict(torch.load(os.path.join(logdir, self.model_name, f'{resume_name}.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(logdir, self.model_name, f'{resume_name}_optim.pth')))
        with open(os.path.join(logdir, self.model_name, 'recent.log'), 'r') as f:
            log = json.load(f)
        self.cur_epoch = log['cur_epoch']
        self.bestval = log['bestval']

    def save(self):
        save_best = False
        if self.val_score[-1] > self.bestval:
            self.bestval = self.val_score[-1]
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'bestval': self.bestval,
            'train_loss': self.train_loss,
            'val_score': self.val_score,
            'train_metrics': ['NLLLoss', 'iou'],
            'val_metrics': ['NLLLoss', 'iou', 'iou_NN'],
        }

        torch.save(model.state_dict(), os.path.join(logdir, self.model_name, 'recent.pth'))
        torch.save(optimizer.state_dict(), os.path.join(logdir, self.model_name, 'recent_optim.pth'))
        # Log other data corresponding to the recent model
        with open(os.path.join(logdir, self.model_name, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))
        tqdm.write('====== Saved recent model ======>')

        if save_best:
            torch.save(model.state_dict(), os.path.join(logdir, self.model_name, 'best.pth'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, self.model_name, 'best_optim.pth'))
            # Log other data corresponding to the recent model
            with open(os.path.join(logdir, self.model_name, 'best.log'), 'w') as f:
                f.write(json.dumps(log_table))
            tqdm.write('====== Overwrote best model ======>')

    def _get_pred(self, data):
        pred_fns = {
            'occ': self._get_pred_occ,
            'residual': self._get_pred_residual,
        }
        return pred_fns[self.model_name](data)

    @staticmethod
    def _get_pred_occ(data):
        inp_odms = extract_odms(data['32'].to(args.device))
        return model(inp_odms)

    @staticmethod
    def _get_pred_residual(data):
        inp_voxels = data['32'].to(args.device)
        inp_odms = extract_odms(inp_voxels)

        initial_odms = upsample_omd(inp_odms) * 4
        distance = 128 - initial_odms
        pred_odms_update = model(inp_odms)
        pred_odms_update = pred_odms_update * distance
        return initial_odms + pred_odms_update

    def _get_loss(self, pred, data):
        tgt_odms = extract_odms(data['128'].to(args.device))
        tgt = to_occupancy_map(tgt_odms) if self.model_name == 'occ' else tgt_odms
        return loss_fn(pred_odms, tgt)

    def _calculate_iou(self, pred_odms, data):
        if self.model_name == 'occ':
            ones = pred_odms > .3
            zeros = pred_odms <= .7
            pred_odms[ones] = pred_odms.shape[-1]
            pred_odms[zeros] = 0
        elif self.model_name == 'residual':
            pred_odms = pred_odms.int()
        else:
            raise ValueError

        tgt_voxels = data['128'].to(args.device)
        inp_voxels = data['32'].to(args.device)
        NN_pred = up_sample(inp_voxels)
        iou_NN = kal.metrics.voxel.iou(NN_pred.contiguous(), tgt_voxels)
        iou_NN_epoch += iou_NN

        pred_voxels = []
        for odms, voxel_NN in zip(pred_odms, NN_pred): 
            pred_voxels.append(kal.rep.voxel.project_odms(odms, voxel_NN, votes=2).unsqueeze(0))
        pred_voxels = torch.cat(pred_voxels)
        return kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels)


for model_name in ['residual', 'occ']:
    # Model
    model = SuperresNetwork(128, 32).to(args.device)
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train
    trainer = Engine(model_name=model_name, resume_name=args.resume)
    print(f'Training {model_name} model...')
    for i, epoch in enumerate(range(args.epochs)): 
        trainer.train()
        if i % args.val_every == 0: 
            trainer.validate()
        trainer.save()
