import argparse
import copy

# from loader import MoleculeDataset
from adv_finetune_fn import MoleculeDataset, scaffold_split, BatchMasking
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from chem_tran_fn import *
from advmask import build_adv
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
# from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

criterion = nn.BCEWithLogitsLoss(reduction = "none")

class GNN_graphpred(nn.Module):
    def __init__(self, args, num_tasks):
        super(GNN_graphpred, self).__init__()
        self.JK = args.JK
        self.emb_dim = args.emb_dim
        self.num_layer = args.num_layer
        self.num_tasks = num_tasks
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.gnn = GNN(args.num_layer, args.emb_dim, args.uniformity_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                gnn_type=args.gnn_type).to(device)
        graph_pooling = args.graph_pooling
        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks).to(device)



    def forward(self, batch, mask_prob, alpha_adv, args):
        node_rep, u_loss = self.gnn(batch, mask_prob, alpha_adv, args)
        return self.graph_pred_linear(self.pool(node_rep, batch.batch)), u_loss


def train(args, model, adv_mask, device, loader, optimizer, adv_optimizer, epoch, num_tasks):
    model.train()
    adv_mask.train()
    alpha_adv = args.alpha_0 + (((epoch - 1) / args.epochs) ** args.gamma) * (args.alpha_T - args.alpha_0)
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        if "masked_atom_indices" not in batch:  ##如果没有提供掩盖的列表则随机选取保存到masked_atom_indices
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = batch.x.size()[0]
            sample_size = int(num_atoms * args.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # print(batch)
        # input(type(batch))

        batch = batch.to(device)
        tmp_batch = BatchMasking()
        tmp_batch.to(device)
        for key in batch:
            tmp_batch[key[0]] = key[1]

        mask_prob = adv_mask(tmp_batch.x, tmp_batch.edge_index, tmp_batch.edge_attr)
        tmp_batch.masked_atom_indices = torch.tensor(masked_atom_indices,device=device)
        tmp_batch.mask_node_label = tmp_batch.x[masked_atom_indices]

        # atom_type = F.one_hot(tmp_batch.mask_node_label[:, 0], num_classes=num_tasks).float()  ##（节点类型，掩码节点数）
        # tmp_batch.node_attr_label = atom_type

        node_rep, u_loss = model(tmp_batch, mask_prob, alpha_adv, args)
        y = batch.y.view(node_rep.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(node_rep.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        adv_optimizer.zero_grad()
        loss_mask = torch.sum(loss_mat) / torch.sum(is_valid)
        loss_mask = -loss_mask + args.belta * (
                torch.tensor([1.]).to(device) / torch.sin(torch.pi / len(batch.x) * (mask_prob[:, 0].sum())))
        loss_mask.backward()
        # for name,para in adv_mask.named_parameters():
        #     print(name)
        #     print(para.grad)
        # input()
        adv_optimizer.step()


        for key in batch:
            tmp_batch[key[0]] = key[1]
        mask_prob = adv_mask(tmp_batch.x, tmp_batch.edge_index, tmp_batch.edge_attr)
        tmp_batch.masked_atom_indices = torch.tensor(masked_atom_indices, device=device)
        tmp_batch.mask_node_label = tmp_batch.x[masked_atom_indices]

        # atom_type = F.one_hot(tmp_batch.mask_node_label[:, 0], num_classes=num_tasks).float()  ##（节点类型，掩码节点数）
        # tmp_batch.node_attr_label = atom_type

        node_rep, u_loss = model(tmp_batch, mask_prob, alpha_adv, args)
        y = batch.y.view(node_rep.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(node_rep.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss = loss + u_loss
        loss.backward()

        # for name,para in model.named_parameters():
        #     print(name)
        #     print(para.grad)
        # input()

        optimizer.step()


def eval(args, model, adv_mask, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        if "masked_atom_indices" not in batch:  ##如果没有提供掩盖的列表则随机选取保存到masked_atom_indices
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = batch.x.size()[0]
            sample_size = int(num_atoms * args.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        tmp_batch = BatchMasking()
        tmp_batch.to(device)
        for key in batch:
            tmp_batch[key[0]] = key[1]
        mask_prob = adv_mask(tmp_batch.x, tmp_batch.edge_index, tmp_batch.edge_attr)
        tmp_batch.masked_atom_indices = torch.tensor(masked_atom_indices, device=device)
        tmp_batch.mask_node_label = tmp_batch.x[masked_atom_indices]
            
        with torch.no_grad():
            # pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            node_rep, u_loss = model(tmp_batch, mask_prob, 0, args)
            
        y_true.append(batch.y.view(node_rep.shape))
        y_scores.append(node_rep)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')

    parser.add_argument('--mask_rate', type=float, default=0.25,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--input_adv_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument("--uniformity_dim", type=int, default=64)  ####还有8，16，128
    parser.add_argument("--alpha_0", type=float, default=0.0)  ####全是0
    parser.add_argument("--alpha_T", type=float, default=0.4)  ####还有0.5，0.6
    parser.add_argument("--gamma", type=float, default=1.0)  ##还有0.8
    parser.add_argument("--lr_mask", type=float, default=0.001, help="mask_learning rate")  ####
    parser.add_argument("--replace_rate", type=float, default=0.1)  ##有0有0.1
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)  ####全是0
    parser.add_argument("--lamda", type=float, default=0.001)  ####还有0.00005
    parser.add_argument("--belta", type=float, default=0.01)  ###还有0.0001，1.0

    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=False)
    args = parser.parse_args()

    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    # elif args.split == "random":
    #     train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random")
    # elif args.split == "random_scaffold":
    #     smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    #     train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)


    #set up model
    # model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    model = GNN(args.num_layer, args.emb_dim, args.uniformity_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                gnn_type=args.gnn_type).to(device)


    model = GNN_graphpred(args, num_tasks)

    adv_mask = build_adv(args)########需要改args

    args.input_model_file = "checkpoints/save_gin.pth"
    if not args.input_model_file == "":
        print("load pretrained model from:", args.input_model_file)
        # model = torch.load(args.input_model_file)
        model.gnn.load_state_dict(torch.load(args.input_model_file))
        # model.from_pretrained(args.input_model_file)

    args.input_adv_model_file = "checkpoints/save_mask_gin.pth"
    if not args.input_adv_model_file == "":
        print("load pretrained model from:", args.input_adv_model_file)
        # adv_mask = torch.load(args.input_adv_model_file)
        adv_mask.load_state_dict(torch.load(args.input_adv_model_file))
        # adv_mask.from_pretrained(args.input_adv_model_file)

    #set up optimizer
    #different learning rate for different part of GNN
    # model_param_group = []
    # model_param_group.append({"params": model.gnn.parameters()})
    # if args.graph_pooling == "attention":
    #     model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)



    adv_optimizer = optim.Adam(adv_mask.parameters(), lr=args.lr_mask, weight_decay=args.decay)

    # print(optimizer)
    # print(adv_optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
        adv_scheduler = torch.optim.lr_scheduler.StepLR(adv_optimizer, step_size=30, gamma=0.3)
    else:
        scheduler = None

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    best_val_acc = 0
    final_test_acc = 0

    result_filename = f"results/{args.dataset}.result"
    fw = open(result_filename, "a")
    fw.write(f"----- seed {args.seed} -------- \n")

    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        train(args, model, adv_mask, device, train_loader, optimizer, adv_optimizer, epoch, num_tasks)
        if scheduler is not None:
            scheduler.step()

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, adv_mask, device, val_loader)
        test_acc = eval(args, model, adv_mask, device, test_loader)

        if args.use_early_stopping and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
        else:
            final_test_acc = test_acc

        print("train: %f val: %f test: %f" %(train_acc, val_acc, final_test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', final_test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()

    fw.write(f"val: {val_acc_list[-1]}, test: {test_acc_list[-1]}\n")
    fw.write(f"\n\n")
    fw.close()

if __name__ == "__main__":
    main()
