from torch_geometric.data import InMemoryDataset
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
import os
from itertools import repeat, product, chain
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import numpy as np
import random
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
import copy

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def neg_sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss



def train_mae(args, model_list, loader, optimizer_list, device, epoch, alpha_l=1.0, loss_fn="sce"):
    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
        mask_criterion = partial(neg_sce_loss, alpha=alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()
        mask_criterion = nn.CrossEntropyLoss()

    model, adv_mask, dec_pred_atoms, dec_pred_bonds = model_list#############
    optimizer_model, optimizer_adv_mask, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds = optimizer_list

    model.train()
    dec_pred_atoms.train()

    if dec_pred_bonds is not None:
        dec_pred_bonds.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    alpha_adv = args.alpha_0 + (((epoch-1) / args.epochs) ** args.gamma) * (args.alpha_T - args.alpha_0)
    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)
        batch_tmp = copy.deepcopy(batch)

        mask_prob = adv_mask(batch.x, batch.edge_index, batch.edge_attr)
        node_rep, u_loss = model(batch, mask_prob, alpha_adv, args)


        ## loss for nodes
        node_attr_label = batch.node_attr_label
        masked_node_indices = batch.masked_atom_indices
        pred_node = dec_pred_atoms(node_rep, batch.edge_index, batch.edge_attr, masked_node_indices)


        # loss = criterion(pred_node.double(), batch.mask_node_label[:,0])
        if loss_fn == "sce":
            # loss = criterion(node_attr_label, pred_node[masked_node_indices])
            loss_mask = mask_criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            loss_mask = criterion(pred_node.double()[masked_node_indices], batch.mask_node_label[:, 0])

        # acc_node = compute_accuracy(pred_node, batch.mask_node_label[:,0])
        # acc_node_accum += acc_node

        if args.mask_edge:
            
            edge_rep = node_rep[batch.edge_index[0]] + node_rep[batch.edge_index[1]]
            pred_edge = dec_pred_bonds(edge_rep, batch.edge_index, batch.edge_attr, batch.connected_edge_indices)
            loss_mask += mask_criterion(pred_edge[batch.connected_edge_indices].double(), batch.edge_attr_label)
            
            # masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            # edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            # pred_edge = dec_pred_bonds(edge_rep, batch.edge_index, batch.edge_attr, batch.connected_edge_indices)
            # # loss += criterion(pred_edge.double(), batch.mask_edge_label[:, 0])
            # loss_mask += criterion(pred_edge.double(), batch.mask_edge_label[:, 0])

            # acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:,0])
            # acc_edge_accum += acc_edge

        loss_mask = -loss_mask + args.belta*(torch.tensor([1.]).to(device)/torch.sin(torch.pi/ len(batch.x) * (mask_prob[:,0].sum())))

        optimizer_model.zero_grad()
        optimizer_adv_mask.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()


        # pred_node.retain_grad()
        # node_rep.retain_grad()
        # mask_prob.retain_grad()

        loss_mask.backward()
        optimizer_adv_mask.step()
        batch = batch_tmp

        optimizer_model.zero_grad()
        optimizer_adv_mask.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()
        mask_prob = adv_mask(batch.x, batch.edge_index, batch.edge_attr)
        node_rep, u_loss = model(batch, mask_prob, alpha_adv, args)
        ## loss for nodes
        node_attr_label = batch.node_attr_label
        masked_node_indices = batch.masked_atom_indices
        pred_node = dec_pred_atoms(node_rep, batch.edge_index, batch.edge_attr, masked_node_indices)

        # loss = criterion(pred_node.double(), batch.mask_node_label[:,0])
        if loss_fn == "sce":
            loss = criterion(node_attr_label, pred_node[masked_node_indices])
            # loss_mask = mask_criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            loss = criterion(pred_node.double()[masked_node_indices], batch.mask_node_label[:, 0])

        # acc_node = compute_accuracy(pred_node, batch.mask_node_label[:,0])
        # acc_node_accum += acc_node

        if args.mask_edge:
            edge_rep = node_rep[batch.edge_index[0]] + node_rep[batch.edge_index[1]]
            pred_edge = dec_pred_bonds(edge_rep, batch.edge_index, batch.edge_attr, batch.connected_edge_indices)
            loss += criterion(pred_edge[batch.connected_edge_indices].double(), batch.edge_attr_label)
            # masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            # edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            # pred_edge = dec_pred_bonds(edge_rep)
            # loss += criterion(pred_edge.double(), batch.mask_edge_label[:, 0])
            # # loss_mask += criterion(pred_edge.double(), batch.mask_edge_label[:, 0])

        loss = loss + u_loss
        loss.backward()

        optimizer_model.step()
        optimizer_dec_pred_atoms.step()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"train_loss: {loss.item():.4f}")

    return loss_accum / step  # , acc_node_accum/step, acc_edge_accum/step

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super().__init__()
        self._dec_type = gnn_type
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr = "add")
        # elif gnn_type == "gcn":
        #     self.conv = GCNConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "linear":
            self.dec = torch.nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_dim]))
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = torch.nn.PReLU()
        self.temp = 0.2


    def forward(self, x, edge_index, edge_attr, mask_node_indices):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.activation(x)
            x = self.enc_to_dec(x)

            x[mask_node_indices] = 0
            # x[mask_node_indices] = self.dec_token
            out = self.conv(x, edge_index, edge_attr)
            # out = F.softmax(out, dim=-1) / self.temp
        return out

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, uniformity_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)


        ##原来使用的是embedding，但这里变成float型，改为linear
        self.x_lin1 = torch.nn.Linear(1,emb_dim)
        self.x_lin2 = torch.nn.Linear(1,emb_dim)
        self.uniformity_dim = uniformity_dim
        if self.JK == "concat":
            self.uniformity_layer = nn.Linear(emb_dim * self.num_layer, self.uniformity_dim, bias=False)
        else:
            self.uniformity_layer = nn.Linear(emb_dim, self.uniformity_dim, bias=False)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    # def forward(self, *argv):
    def forward(self, batch, mask_prob, alpha_adv, args):
        # batch, mask_prob, alpha_adv, args = argv[0], argv[1], argv[2], argv[3]
        # input(batch)
        # if len(argv) == 3:
        #     batch, mask_prob, alpha_adv, args = argv[0], argv[1], argv[2], argv[3]
        # elif len(argv) == 1:
        #     data = argv[0]
        #     x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # else:
        #     raise ValueError("unmatched number of arguments.")
        
        x = batch.x

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        
        
        # print(batch)
        # print("pre")

        num_nodes = len(x)
        mask_num_nodes = len(batch.masked_atom_indices)
        num_random_mask_nodes = int(mask_num_nodes * (1. - alpha_adv))


        random_mask_nodes = batch.masked_atom_indices[:num_random_mask_nodes]
        random_keep_nodes = batch.masked_atom_indices[num_random_mask_nodes:]

        mask_ = mask_prob[:, 1]
        perm_adv = torch.randperm(num_nodes, device=x.device)
        adv_keep_token = perm_adv[:int(num_nodes * (1. - alpha_adv))]
        mask_[adv_keep_token] = 1.
        Mask_ = mask_.reshape(-1, 1)
        adv_keep_nodes = mask_.nonzero().reshape(-1)
        adv_mask_nodes = (1 - mask_).nonzero().reshape(-1)


        mask_nodes = torch.cat((random_mask_nodes, adv_mask_nodes), dim=0).unique()
        keep_nodes = torch.tensor(np.intersect1d(random_keep_nodes.cpu().numpy(), adv_keep_nodes.cpu().numpy())).to(x.device)
        num_mask_nodes = mask_nodes.shape[0]

        if args.replace_rate > 0:
            num_noise_nodes = int(args.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int( ( 1 - args.replace_rate ) * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(args.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            one = torch.ones_like(x, dtype = torch.float)
            # one = torch.ones_like(x, dtype = torch.long)
            x = x * one
            out_x = out_x * Mask_
            out_x[token_nodes] = torch.tensor([119.0, 0]).to(x.device)
            # out_x[token_nodes] = torch.tensor([119, 0]).to(x.device)
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            out_x = out_x * Mask_
            token_nodes = mask_nodes
            out_x[token_nodes] = torch.tensor([119.0, 0]).to(x.device)
            # out_x[token_nodes] = torch.tensor([119, 0]).to(x.device)

        x = out_x

        batch.mask_node_label = batch.x[mask_nodes]
        atom_type = F.one_hot(batch.mask_node_label[:, 0], num_classes=119).float()  ##（节点类型，掩码节点数）
        batch.node_attr_label = atom_type  ##（节点类型，掩码节点数），类型表示为独热编码
        batch.x = x
        
        batch.masked_atom_indices = mask_nodes

        mask_nodes_tmp = copy.deepcopy(mask_nodes).to(torch.device("cpu"))

        if args.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms

            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(edge_index.cpu().numpy().T):
                for atom_idx in mask_nodes_tmp:#################
                    if atom_idx.item() in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            print(len(connected_edge_indices))
            connected_edge_indices = connected_edge_indices
            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(edge_attr[bond_idx].view(1, -1))

                batch.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)

                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    batch.edge_attr[bond_idx] = torch.tensor([5, 0])

                batch.connected_edge_indices = torch.tensor(connected_edge_indices[::2])
            else:
                batch.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                batch.connected_edge_indices = torch.tensor(connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(batch.mask_edge_label[:, 0], num_classes=5).float()
            bond_direction = F.one_hot(batch.mask_edge_label[:, 1], num_classes=3).float()
            batch.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)

        if args.drop_edge_rate > 0:
            pass

        # print(batch)
        # input("last")


        edge_attr = batch.edge_attr
        # x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1]) ##########
        x = self.x_lin1(x[:, 0].reshape(-1,1)) + self.x_lin2(x[:, 1].reshape(-1,1))
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        lamda = args.lamda * (1.0 - alpha_adv)
        node_eb = F.relu(self.uniformity_layer(node_representation))
        u_loss = uniformity_loss(node_eb, lamda)


        return node_representation, u_loss



def uniformity_loss(node_rep, t, max_size=30000, batch=10000):
    # calculate loss
    n = node_rep.size(0)
    node_rep = torch.nn.functional.normalize(node_rep)
    if n < max_size:
       loss = torch.log(torch.exp(2. * t * ((node_rep @ node_rep.T) - 1.)).mean())
    else:
        total_loss = 0.
        permutation = torch.randperm(n)
        node_rep = node_rep[permutation]
        for i in range(0, n, batch):
            batch_features = node_rep[i:i + batch]
            batch_loss = torch.log(torch.exp(2. * t * ((batch_features @ batch_features.T) - 1.)).mean())
            total_loss += batch_loss
        loss = total_loss / (n // batch)
    return loss


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

# def generate_scaffold(smiles, include_chirality=False):
#     """
#     Obtain Bemis-Murcko scaffold from smiles
#     :param smiles:
#     :param include_chirality:
#     :return: smiles of scaffold
#     """
#     scaffold = MurckoScaffold.MurckoScaffoldSmiles(
#         smiles=smiles, includeChirality=include_chirality)
#     return scaffold
#
# def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
#                    frac_train=0.8, frac_valid=0.1, frac_test=0.1,
#                    return_smiles=False):
#     """
#     Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
#     Split dataset by Bemis-Murcko scaffolds
#     This function can also ignore examples containing null values for a
#     selected task when splitting. Deterministic split
#     :param dataset: pytorch geometric dataset obj
#     :param smiles_list: list of smiles corresponding to the dataset obj
#     :param task_idx: column idx of the data.y tensor. Will filter out
#     examples with null value in specified task column of the data.y tensor
#     prior to splitting. If None, then no filtering
#     :param null_value: float that specifies null value in data.y to filter if
#     task_idx is provided
#     :param frac_train:
#     :param frac_valid:
#     :param frac_test:
#     :param return_smiles:
#     :return: train, valid, test slices of the input dataset obj. If
#     return_smiles = True, also returns ([train_smiles_list],
#     [valid_smiles_list], [test_smiles_list])
#     """
#     np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
#
#     if task_idx != None:
#         # filter based on null values in task_idx
#         # get task array
#         y_task = np.array([data.y[task_idx].item() for data in dataset])
#         # boolean array that correspond to non null values
#         non_null = y_task != null_value
#         smiles_list = list(compress(enumerate(smiles_list), non_null))
#     else:
#         non_null = np.ones(len(dataset)) == 1
#         smiles_list = list(compress(enumerate(smiles_list), non_null))
#
#     # create dict of the form {scaffold_i: [idx1, idx....]}
#     all_scaffolds = {}
#     for i, smiles in smiles_list:
#         scaffold = generate_scaffold(smiles, include_chirality=True)
#         if scaffold not in all_scaffolds:
#             all_scaffolds[scaffold] = [i]
#         else:
#             all_scaffolds[scaffold].append(i)
#
#     # sort from largest to smallest sets
#     all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
#     all_scaffold_sets = [
#         scaffold_set for (scaffold, scaffold_set) in sorted(
#             all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
#     ]
#
#     # get train, valid test indices
#     train_cutoff = frac_train * len(smiles_list)
#     valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
#     train_idx, valid_idx, test_idx = [], [], []
#     for scaffold_set in all_scaffold_sets:
#         if len(train_idx) + len(scaffold_set) > train_cutoff:
#             if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
#                 test_idx.extend(scaffold_set)
#             else:
#                 valid_idx.extend(scaffold_set)
#         else:
#             train_idx.extend(scaffold_set)
#
#     assert len(set(train_idx).intersection(set(valid_idx))) == 0
#     assert len(set(test_idx).intersection(set(valid_idx))) == 0
#
#     train_dataset = dataset[torch.tensor(train_idx)]
#     valid_dataset = dataset[torch.tensor(valid_idx)]
#     test_dataset = dataset[torch.tensor(test_idx)]
#
#     if not return_smiles:
#         return train_dataset, valid_dataset, test_dataset
#     else:
#         train_smiles = [smiles_list[i][1] for i in train_idx]
#         valid_smiles = [smiles_list[i][1] for i in valid_idx]
#         test_smiles = [smiles_list[i][1] for i in test_idx]
#         return train_dataset, valid_dataset, test_dataset, (train_smiles,
#                                                             valid_smiles,
#                                                             test_smiles)

class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

        self.num_chirality_tag = 3
        self.num_bond_direction = 3

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:  ##如果没有提供掩盖的列表则随机选取保存到masked_atom_indices
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))


        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)  ##被掩盖的原子特征保存到mask_node_label
        data.masked_atom_indices = torch.tensor(masked_atom_indices)    ##被掩盖的原子索引保存到masked_atom_indices

        # ----------- graphMAE -----------
        atom_type = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()   ##（节点类型，掩码节点数）
        atom_chirality = F.one_hot(data.mask_node_label[:, 1], num_classes=self.num_chirality_tag).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        data.node_attr_label = atom_type        ##（节点类型，掩码节点数），类型表示为独热编码





        # modify the original node feature of the masked node
        # for atom_idx in masked_atom_indices:
        #     data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])  ##掩盖的节点特征改为（num_atom_type, 0）





        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)





                # modify the original bond features of the bonds connected to the mask atoms
                # for bond_idx in connected_edge_indices:
                #     data.edge_attr[bond_idx] = torch.tensor(
                #         [self.num_edge_type, 0])




                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)
            # data.edge_attr_label = edge_type

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys


        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []



        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                elif key  == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)


        # print(batch)
        # print(batch.x)

        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class DataLoaderMaskingPred(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, mask_rate=0.0, mask_edge=0.0, **kwargs):
        self._transform = MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=mask_rate, mask_edge=mask_edge)
        super(DataLoaderMaskingPred, self).__init__(dataset,batch_size,shuffle,collate_fn=self.collate_fn,**kwargs)

    def collate_fn(self, batches):
        batchs = [self._transform(x) for x in batches]
        return BatchMasking.from_data_list(batchs)

class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 # data = None,
                 # slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):

        # print(self.raw_dir)
        # input()
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset == 'zinc_standard_agent':
            input_path = self.raw_paths[0]
            input_df = pd.read_csv(input_path, sep=',', compression='gzip',
                                   dtype='str')
            smiles_list = list(input_df['smiles'])
            zinc_id_list = list(input_df['zinc_id'])
            for i in range(len(smiles_list)):
                s = smiles_list[i]
                # each example contains a single species
                try:
                    rdkit_mol = AllChem.MolFromSmiles(s)
                    if rdkit_mol != None:  # ignore invalid mol objects
                        # # convert aromatic bonds to double bonds
                        # Chem.SanitizeMol(rdkit_mol,
                        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                        data = mol_to_graph_data_obj_simple(rdkit_mol)
                        # manually add mol id
                        id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                        data.id = torch.tensor(
                            [id])  # id here is zinc id value, stripped of
                        # leading zeros
                        data_list.append(data)
                        data_smiles_list.append(smiles_list[i])
                except:
                    continue

        # elif self.dataset == "zinc_sample":
        #     input_path = self.raw_paths[0]
        #     with open(input_path, "r") as f:
        #         data = f.readlines()
        #     all_data = [x.strip() for x in data]
        #     data_smiles_list = []
        #     data_list = []
        #     for i, item in enumerate(all_data):
        #         s = item
        #         try:
        #             rdkit_mol = AllChem.MolFromSmiles(s)
        #             if rdkit_mol != None:
        #                 data = mol_to_graph_data_obj_simple(rdkit_mol)
        #                 # manually add mol id
        #                 id = i
        #                 data.id = torch.tensor([id])  # id here is zinc id value, stripped of
        #                 # leading zeros
        #                 data_list.append(data)
        #                 data_smiles_list.append(s)
        #         except:
        #             continue
        #
        # elif self.dataset == 'chembl_filtered':
        #     ### get downstream test molecules.
        #     # from splitters import scaffold_split
        #
        #     ###
        #     downstream_dir = [
        #         'dataset/bace',
        #         'dataset/bbbp',
        #         'dataset/clintox',
        #         'dataset/esol',
        #         'dataset/freesolv',
        #         'dataset/hiv',
        #         'dataset/lipophilicity',
        #         'dataset/muv',
        #         # 'dataset/pcba/processed/smiles.csv',
        #         'dataset/sider',
        #         'dataset/tox21',
        #         'dataset/toxcast'
        #     ]
        #
        #     downstream_inchi_set = set()
        #     for d_path in downstream_dir:
        #         print(d_path)
        #         dataset_name = d_path.split('/')[1]
        #         downstream_dataset = MoleculeDataset(d_path, dataset=dataset_name)
        #         downstream_smiles = pd.read_csv(os.path.join(d_path,
        #                                                      'processed', 'smiles.csv'),
        #                                         header=None)[0].tolist()
        #
        #         assert len(downstream_dataset) == len(downstream_smiles)
        #
        #         _, _, _, (train_smiles, valid_smiles, test_smiles) = scaffold_split(downstream_dataset,
        #                                                                             downstream_smiles, task_idx=None,
        #                                                                             null_value=0,
        #                                                                             frac_train=0.8, frac_valid=0.1,
        #                                                                             frac_test=0.1,
        #                                                                             return_smiles=True)
        #
        #         ### remove both test and validation molecules
        #         remove_smiles = test_smiles + valid_smiles
        #
        #         downstream_inchis = []
        #         for smiles in remove_smiles:
        #             species_list = smiles.split('.')
        #             for s in species_list:  # record inchi for all species, not just
        #                 # largest (by default in create_standardized_mol_id if input has
        #                 # multiple species)
        #                 inchi = create_standardized_mol_id(s)
        #                 downstream_inchis.append(inchi)
        #         downstream_inchi_set.update(downstream_inchis)
        #
        #     smiles_list, rdkit_mol_objs, folds, labels = \
        #         _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))
        #
        #     print('processing')
        #     for i in range(len(rdkit_mol_objs)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         if rdkit_mol != None:
        #             # # convert aromatic bonds to double bonds
        #             # Chem.SanitizeMol(rdkit_mol,
        #             #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #             mw = Descriptors.MolWt(rdkit_mol)
        #             if 50 <= mw <= 900:
        #                 inchi = create_standardized_mol_id(smiles_list[i])
        #                 if inchi != None and inchi not in downstream_inchi_set:
        #                     data = mol_to_graph_data_obj_simple(rdkit_mol)
        #                     # manually add mol id
        #                     data.id = torch.tensor(
        #                         [i])  # id here is the index of the mol in
        #                     # the dataset
        #                     data.y = torch.tensor(labels[i, :])
        #                     # fold information
        #                     if i in folds[0]:
        #                         data.fold = torch.tensor([0])
        #                     elif i in folds[1]:
        #                         data.fold = torch.tensor([1])
        #                     else:
        #                         data.fold = torch.tensor([2])
        #                     data_list.append(data)
        #                     data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'tox21':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_tox21_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         ## convert aromatic bonds to double bonds
        #         # Chem.SanitizeMol(rdkit_mol,
        #         # sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #         data = mol_to_graph_data_obj_simple(rdkit_mol)
        #         # manually add mol id
        #         data.id = torch.tensor(
        #             [i])  # id here is the index of the mol in
        #         # the dataset
        #         data.y = torch.tensor(labels[i, :])
        #         data_list.append(data)
        #         data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'hiv':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_hiv_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         # # convert aromatic bonds to double bonds
        #         # Chem.SanitizeMol(rdkit_mol,
        #         #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #         data = mol_to_graph_data_obj_simple(rdkit_mol)
        #         # manually add mol id
        #         data.id = torch.tensor(
        #             [i])  # id here is the index of the mol in
        #         # the dataset
        #         data.y = torch.tensor([labels[i]])
        #         data_list.append(data)
        #         data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'bace':
        #     smiles_list, rdkit_mol_objs, folds, labels = \
        #         _load_bace_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         # # convert aromatic bonds to double bonds
        #         # Chem.SanitizeMol(rdkit_mol,
        #         #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #         data = mol_to_graph_data_obj_simple(rdkit_mol)
        #         # manually add mol id
        #         data.id = torch.tensor(
        #             [i])  # id here is the index of the mol in
        #         # the dataset
        #         data.y = torch.tensor([labels[i]])
        #         data.fold = torch.tensor([folds[i]])
        #         data_list.append(data)
        #         data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'bbbp':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_bbbp_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         if rdkit_mol != None:
        #             # # convert aromatic bonds to double bonds
        #             # Chem.SanitizeMol(rdkit_mol,
        #             #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #             data = mol_to_graph_data_obj_simple(rdkit_mol)
        #             # manually add mol id
        #             data.id = torch.tensor(
        #                 [i])  # id here is the index of the mol in
        #             # the dataset
        #             data.y = torch.tensor([labels[i]])
        #             data_list.append(data)
        #             data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'clintox':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_clintox_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         if rdkit_mol != None:
        #             # # convert aromatic bonds to double bonds
        #             # Chem.SanitizeMol(rdkit_mol,
        #             #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #             data = mol_to_graph_data_obj_simple(rdkit_mol)
        #             # manually add mol id
        #             data.id = torch.tensor(
        #                 [i])  # id here is the index of the mol in
        #             # the dataset
        #             data.y = torch.tensor(labels[i, :])
        #             data_list.append(data)
        #             data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'esol':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_esol_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         # # convert aromatic bonds to double bonds
        #         # Chem.SanitizeMol(rdkit_mol,
        #         #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #         data = mol_to_graph_data_obj_simple(rdkit_mol)
        #         # manually add mol id
        #         data.id = torch.tensor(
        #             [i])  # id here is the index of the mol in
        #         # the dataset
        #         data.y = torch.tensor([labels[i]])
        #         data_list.append(data)
        #         data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'freesolv':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_freesolv_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         # # convert aromatic bonds to double bonds
        #         # Chem.SanitizeMol(rdkit_mol,
        #         #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #         data = mol_to_graph_data_obj_simple(rdkit_mol)
        #         # manually add mol id
        #         data.id = torch.tensor(
        #             [i])  # id here is the index of the mol in
        #         # the dataset
        #         data.y = torch.tensor([labels[i]])
        #         data_list.append(data)
        #         data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'lipophilicity':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_lipophilicity_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         # # convert aromatic bonds to double bonds
        #         # Chem.SanitizeMol(rdkit_mol,
        #         #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #         data = mol_to_graph_data_obj_simple(rdkit_mol)
        #         # manually add mol id
        #         data.id = torch.tensor(
        #             [i])  # id here is the index of the mol in
        #         # the dataset
        #         data.y = torch.tensor([labels[i]])
        #         data_list.append(data)
        #         data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'muv':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_muv_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         # # convert aromatic bonds to double bonds
        #         # Chem.SanitizeMol(rdkit_mol,
        #         #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #         data = mol_to_graph_data_obj_simple(rdkit_mol)
        #         # manually add mol id
        #         data.id = torch.tensor(
        #             [i])  # id here is the index of the mol in
        #         # the dataset
        #         data.y = torch.tensor(labels[i, :])
        #         data_list.append(data)
        #         data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'pcba':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_pcba_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         # # convert aromatic bonds to double bonds
        #         # Chem.SanitizeMol(rdkit_mol,
        #         #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #         data = mol_to_graph_data_obj_simple(rdkit_mol)
        #         # manually add mol id
        #         data.id = torch.tensor(
        #             [i])  # id here is the index of the mol in
        #         # the dataset
        #         data.y = torch.tensor(labels[i, :])
        #         data_list.append(data)
        #         data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'pcba_pretrain':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_pcba_dataset(self.raw_paths[0])
        #     downstream_inchi = set(pd.read_csv(os.path.join(self.root,
        #                                                     'downstream_mol_inchi_may_24_2019'),
        #                                        sep=',', header=None)[0])
        #     for i in range(len(smiles_list)):
        #         if '.' not in smiles_list[i]:  # remove examples with
        #             # multiples species
        #             rdkit_mol = rdkit_mol_objs[i]
        #             mw = Descriptors.MolWt(rdkit_mol)
        #             if 50 <= mw <= 900:
        #                 inchi = create_standardized_mol_id(smiles_list[i])
        #                 if inchi != None and inchi not in downstream_inchi:
        #                     # # convert aromatic bonds to double bonds
        #                     # Chem.SanitizeMol(rdkit_mol,
        #                     #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #                     data = mol_to_graph_data_obj_simple(rdkit_mol)
        #                     # manually add mol id
        #                     data.id = torch.tensor(
        #                         [i])  # id here is the index of the mol in
        #                     # the dataset
        #                     data.y = torch.tensor(labels[i, :])
        #                     data_list.append(data)
        #                     data_smiles_list.append(smiles_list[i])
        #
        # # elif self.dataset == ''
        #
        # elif self.dataset == 'sider':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_sider_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         # # convert aromatic bonds to double bonds
        #         # Chem.SanitizeMol(rdkit_mol,
        #         #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #         data = mol_to_graph_data_obj_simple(rdkit_mol)
        #         # manually add mol id
        #         data.id = torch.tensor(
        #             [i])  # id here is the index of the mol in
        #         # the dataset
        #         data.y = torch.tensor(labels[i, :])
        #         data_list.append(data)
        #         data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'toxcast':
        #     smiles_list, rdkit_mol_objs, labels = \
        #         _load_toxcast_dataset(self.raw_paths[0])
        #     for i in range(len(smiles_list)):
        #         rdkit_mol = rdkit_mol_objs[i]
        #         if rdkit_mol != None:
        #             # # convert aromatic bonds to double bonds
        #             # Chem.SanitizeMol(rdkit_mol,
        #             #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #             data = mol_to_graph_data_obj_simple(rdkit_mol)
        #             # manually add mol id
        #             data.id = torch.tensor(
        #                 [i])  # id here is the index of the mol in
        #             # the dataset
        #             data.y = torch.tensor(labels[i, :])
        #             data_list.append(data)
        #             data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'ptc_mr':
        #     input_path = self.raw_paths[0]
        #     input_df = pd.read_csv(input_path, sep=',', header=None, names=['id', 'label', 'smiles'])
        #     smiles_list = input_df['smiles']
        #     labels = input_df['label'].values
        #     for i in range(len(smiles_list)):
        #         s = smiles_list[i]
        #         rdkit_mol = AllChem.MolFromSmiles(s)
        #         if rdkit_mol != None:  # ignore invalid mol objects
        #             # # convert aromatic bonds to double bonds
        #             # Chem.SanitizeMol(rdkit_mol,
        #             #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #             data = mol_to_graph_data_obj_simple(rdkit_mol)
        #             # manually add mol id
        #             data.id = torch.tensor(
        #                 [i])
        #             data.y = torch.tensor([labels[i]])
        #             data_list.append(data)
        #             data_smiles_list.append(smiles_list[i])
        #
        # elif self.dataset == 'mutag':
        #     smiles_path = os.path.join(self.root, 'raw', 'mutag_188_data.can')
        #     # smiles_path = 'dataset/mutag/raw/mutag_188_data.can'
        #     labels_path = os.path.join(self.root, 'raw', 'mutag_188_target.txt')
        #     # labels_path = 'dataset/mutag/raw/mutag_188_target.txt'
        #     smiles_list = pd.read_csv(smiles_path, sep=' ', header=None)[0]
        #     labels = pd.read_csv(labels_path, header=None)[0].values
        #     for i in range(len(smiles_list)):
        #         s = smiles_list[i]
        #         rdkit_mol = AllChem.MolFromSmiles(s)
        #         if rdkit_mol != None:  # ignore invalid mol objects
        #             # # convert aromatic bonds to double bonds
        #             # Chem.SanitizeMol(rdkit_mol,
        #             #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #             data = mol_to_graph_data_obj_simple(rdkit_mol)
        #             # manually add mol id
        #             data.id = torch.tensor(
        #                 [i])
        #             data.y = torch.tensor([labels[i]])
        #             data_list.append(data)
        #             data_smiles_list.append(smiles_list[i])


        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

