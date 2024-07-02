import argparse
import torch.optim as optim
import torch
import numpy as np
from chem_tran_fn import *
from advmask import build_adv
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.25,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = 'save', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=8, help = "Seed for splitting dataset.")
    # parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--use_scheduler", action="store_true", default=False)

    parser.add_argument("--alpha_0", type=float, default=0.0)
    parser.add_argument("--alpha_T", type=float, default=0.4)# 0.5，0.6
    parser.add_argument("--gamma", type=float, default=1.0)## 0.8
    parser.add_argument("--replace_rate", type=float, default=0.1)# 0，0.1
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--lamda", type=float, default=0.001)# 0.00005
    parser.add_argument("--uniformity_dim", type=int, default=64)# 8，16，128
    parser.add_argument("--belta", type=float, default=0.01)# 0.0001，1.0
    parser.add_argument("--lr_mask", type=float, default=0.001, help="mask_learning rate")


    # parser.add_argument("--norm", type=str, default=None)
    # parser.add_argument("--residual", action="store_true", default=False,
    #                     help="use residual connection")
    # parser.add_argument("--activation", type=str, default="prelu")
    # parser.add_argument("--num_hidden", type=int, default=256,
    #                     help="number of hidden units")
    # parser.add_argument("--in_drop", type=float, default=.2,
    #                     help="input feature dropout")


    # parser.add_argument("--alpha_l", type=float, default=2, help="`pow`coefficient for `sce` loss")
    args = parser.parse_args()

    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.mask_edge))

    dataset_name = args.dataset
    dataset = MoleculeDataset("dataset/" + dataset_name, dataset=dataset_name)
    # print(dataset)

    loader = DataLoaderMaskingPred(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   mask_rate=args.mask_rate, mask_edge=args.mask_edge)


    # for x in loader:
    #     print(x)
    #     # for x in x.masked_atom_indices:
    #     #     print(x)
    #     break

    model = GNN(args.num_layer, args.emb_dim, args.uniformity_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)

    adv_mask = build_adv(args)

    if args.input_model_file is not None and args.input_model_file != "":
        model.load_state_dict(torch.load(args.input_model_file))
        print("Resume training from:", args.input_model_file)
        resume = True
    else:
        resume = False

    NUM_NODE_ATTR = 119  # + 3
    atom_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
    if args.mask_edge:
        NUM_BOND_ATTR = 5 + 3
        bond_pred_decoder = GNNDecoder(args.emb_dim, NUM_BOND_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None
    model_list = [model, adv_mask, atom_pred_decoder, bond_pred_decoder]#############
    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_adv_mask = optim.Adam(adv_mask.parameters(), lr=args.lr_mask, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.use_scheduler:
        print("--------- Use scheduler -----------")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
        scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=scheduler)
        scheduler_adv_mask = torch.optim.lr_scheduler.LambdaLR(optimizer_adv_mask, lr_lambda=scheduler)
        scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        scheduler_list = [scheduler_model, scheduler_dec, None]
    else:
        scheduler_model = None
        scheduler_dec = None

    optimizer_list = [optimizer_model, optimizer_adv_mask, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds]
    output_file_temp = "./checkpoints/" + args.output_model_file + f"_{args.gnn_type}"
    output_file_mask_temp = "./checkpoints/" + args.output_model_file + "_mask" + f"_{args.gnn_type}"
    output_file_atom_decoder_temp = "./checkpoints/" + args.output_model_file + "_atom_decoder" + f"_{args.gnn_type}"
    output_file_bond_decoder_temp = "./checkpoints/" + args.output_model_file + "_bond_decoder" + f"_{args.gnn_type}"

    print(resume)
    if not args.output_model_file == "":
        print("save")
    else:
        print("not save")


    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        # train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device)
        # print(train_loss, train_acc_atom, train_acc_bond)

        train_loss = train_mae(args, model_list, loader, optimizer_list, device, epoch, alpha_l=args.alpha_l,
                               loss_fn=args.loss_fn)
        if not resume:
            if epoch % 10 == 0:
                torch.save(model.state_dict(), output_file_temp + f"_{epoch}.pth")
                torch.save(adv_mask.state_dict(), output_file_mask_temp + f"_{epoch}.pth")
                torch.save(atom_pred_decoder.state_dict(), output_file_atom_decoder_temp + f"_{epoch}.pth")
                if args.mask_edge:
                    torch.save(bond_pred_decoder.state_dict(), output_file_bond_decoder_temp + f"_{epoch}.pth")

        print(train_loss)
        if scheduler_model is not None:
            scheduler_model.step()
        if scheduler_dec is not None:
            scheduler_dec.step()

    output_file = "./checkpoints/" + args.output_model_file + f"_{args.gnn_type}"
    output_file_mask = "./checkpoints/" + args.output_model_file + "_mask" + f"_{args.gnn_type}"
    output_file_atom_decoder = "./checkpoints/" + args.output_model_file + "_atom_decoder" + f"_{args.gnn_type}"
    output_file_bond_decoder = "./checkpoints/" + args.output_model_file + "_bond_decoder" + f"_{args.gnn_type}"
    if resume:
        torch.save(model.state_dict(), args.input_model_file.rsplit(".", 1)[0] + f"_resume_{args.epochs}.pth")
    elif not args.output_model_file == "":
        torch.save(model.state_dict(), output_file + ".pth")
        torch.save(adv_mask.state_dict(), output_file_mask + ".pth")
        torch.save(atom_pred_decoder.state_dict(), output_file_atom_decoder + ".pth")
        if args.mask_edge:
            torch.save(bond_pred_decoder.state_dict(), output_file_bond_decoder + ".pth")

if __name__ == "__main__":
    main()