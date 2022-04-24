import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import sys
import numpy as np, argparse, time, pickle, random, itertools, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from graphmask_probe import GraphMaskProbe
from tools.utils.torch_utils.lagrangian_optimization import LagrangianOptimization
from tools.utils.moving_average import MovingAverage

# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        # import ipdb;ipdb.set_trace()
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]        
        max_sequence_len.append(textf.size(0))
        
        # log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, acouf, visuf), dim=-1), qmask, umask)
        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask) # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    #if torch.cuda.is_available():
    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)

        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el


def overwrite_labels(label, log_prob, lengths):
    new_label = torch.argmax(log_prob, 1)
    return new_label


def train_or_eval_graphmask(probe, model, loss_function, data, gnn, cuda, train=False):
    model.eval()
    textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
    lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
    log_prob, _, _, _, _ = model(textf, qmask, umask, lengths)
    # lp = log_prob

    # if train: 
    #     label = overwrite_labels(label, log_prob, lengths)
    #     probe.train()
    # else: 
    #     label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
    #     probe.eval()

    label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
    if train:
        probe.train()
    else:
        probe.eval()

    ori_pred = torch.argmax(log_prob, 1).cpu().numpy()
    ori_loss = loss_function(log_prob, label)

    # model.train()
    gates, baselines, penalty = probe(gnn)
    gnn.inject_message_scale(gates)
    gnn.inject_message_replacement(baselines)
    log_prob, _, _, _, _ = model(textf, qmask, umask, lengths)
    loss = loss_function(log_prob, label)
    # loss = loss_function(lp, label)
    pred = torch.argmax(log_prob, 1).cpu().numpy()
    label = label.cpu().numpy()
    return torch.abs(loss-ori_loss), penalty, pred, label, gates, ori_loss, ori_pred


def initialise_for_model(model):
    vertex_embedding_dims = np.array([model.graph_net.conv1.in_channels, model.graph_net.conv2.in_channels])
    message_dims = np.array([model.graph_net.conv1.out_channels, model.graph_net.conv2.out_channels])
    probe = GraphMaskProbe(vertex_embedding_dims, message_dims, message_dims)
    return probe


def disable_all_gradients(module):
    for param in module.parameters():
        param.requires_grad = False


def validate(probe, model, loss_function, data_loader, gnn, cuda, split="dev"):
    losses, preds, labels, ori_losses, ori_preds = [], [], [], [], []
    scores, vids = [], []
    all_gates = 0
    all_messages = 0
    layer0_gates = 0
    layer0_messages = 0
    layer1_gates = 0
    layer1_messages = 0

    with torch.no_grad():
        for data in data_loader:
            loss, penalty, pred, label, gates, ori_loss, ori_pred = train_or_eval_graphmask(probe, model, loss_function, data, gnn, cuda)
            loss = loss.item()
            ori_loss = ori_loss.item()
            all_gates += float(sum([g.sum().detach().cpu() for g in gates]))
            all_messages += float(gnn.count_latest_messages())
            layer0_gates += float(gates[0].sum().detach().cpu())
            layer1_gates += float(gates[1].sum().detach().cpu())
            layer0_messages += float(gnn.get_latest_messages()[0].numel()/gnn.get_latest_messages()[0].shape[-1])
            layer1_messages += float(gnn.get_latest_messages()[1].numel()/gnn.get_latest_messages()[1].shape[-1])
            preds.append(pred)
            labels.append(label)
            losses.append(loss)
            ori_preds.append(ori_pred)
            ori_losses.append(ori_loss)

    if preds!=[]:
        preds  = np.concatenate(preds)
        ori_preds = np.concatenate(ori_preds)
        labels = np.concatenate(labels)
    else:
        return None, None

    labels = np.array(labels)
    preds = np.array(preds)
    losses = np.array(losses)
    ori_preds = np.array(ori_preds)
    ori_losses = np.array(ori_losses)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds), 4)
    avg_fscore = round(f1_score(labels, preds, average='weighted'), 4)
    avg_ori_loss = round(np.sum(ori_losses)/len(ori_losses), 4)
    avg_ori_accuracy = round(accuracy_score(labels, ori_preds), 4)
    avg_ori_fscore = round(f1_score(labels, ori_preds, average='weighted'), 4)

    print("GraphMask comparison on the "+split+"-split:")
    print("======================================")
    print("Original average fscore: " + str(avg_ori_fscore))
    print("Gated average fscore: " + str(avg_fscore))
    print("Retained messages: all: {}, layer0: {}, layer1: {}".format(
        str(all_gates / all_messages), 
        str(layer0_gates / layer0_messages),
        str(layer1_gates / layer1_messages)
    ))

    sparsity = float(all_gates / all_messages)

    diff = np.abs(avg_ori_fscore - avg_fscore)
    percent_div = float(diff / (avg_ori_fscore + 1e-8))

    if split == "test":
        pass

    return percent_div, sparsity


if __name__ == '__main__':

    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.0, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    
    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
    
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--load', action='store_true', default=False, help='whether to load the pretrained model')
    
    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    n_classes  = 6
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    load       = args.load
    #load       = True

    D_m = 100   # utterance feature dimension
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100

    if args.graph_model:
        seed_everything()
        model = DialogueGCNModel(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=2,
                                 max_seq_len=110,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda)

        print ('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'DialogRNN':
            model = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a, 
                                   n_classes=n_classes,
                                   listener_state=args.active_listener,
                                   context_attention=args.attention,
                                   dropout_rec=args.rec_dropout,
                                   dropout=args.dropout)

            print ('Basic Dialog RNN Model.')


        elif args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic LSTM Model.')

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()

    loss_weights = torch.FloatTensor([1/0.086747,
                                      1/0.144406,
                                      1/0.227883,
                                      1/0.160585,
                                      1/0.127711,
                                      1/0.252668])
    
    if args.class_weight:
        if args.graph_model:
            loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        if args.graph_model:
            loss_function = nn.NLLLoss()
        else:
            loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                  batch_size=batch_size,
                                                                  num_workers=0)

    if load == False:
        best_fscore, best_loss, best_label, best_pred, best_mask = 0, 0, 0, 0, 0
        all_fscore, all_acc, all_loss = [], [], []

        for e in range(n_epochs):
            start_time = time.time()

            if args.graph_model:
                train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, train_loader, e, cuda, optimizer, True)
                valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, valid_loader, e, cuda)
                test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda)
                all_fscore.append(test_fscore)
                all_acc.append(test_acc)

                if test_fscore > best_fscore: 
                    best_fscore = test_fscore
                    torch.save(model.state_dict(), './dgcn_model.pkl')


            else:
                train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
                valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
                test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)
                all_fscore.append(test_fscore)
                all_acc.append(test_acc)
                # torch.save({'model_state_dict': model.state_dict()}, path + name + args.base_model + '_' + str(e) + '.pkl')

            if args.tensorboard:
                writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
                writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)

            print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                    format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        

        if args.tensorboard:
            writer.close()

        print('Test performance..')
        print('Test-acc:', max(all_acc))
        print('F-Score:', max(all_fscore))


    model.load_state_dict(torch.load('./dgcn_model.pkl'))
    test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, 60, cuda)
    print("Original average fscore: " + str(test_fscore))

    # ================Analysing========================
    print("Analyser start...")

    best_sparsity = 1.01
    allowance = 0.03
    divergence_scaling = 5
    penalty_scaling = 1.5
    max_allowed_performance_diff = 0.05
    save_path = "/dgcn_model_probe"
    best_path = None
    seed_everything()
    learning_rate = 3e-4
    test_every_n = 1
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.1,
                                                                batch_size=batch_size,
                                                                num_workers=0)

    # Initialising
    probe = initialise_for_model(model)

    # Make pretrained model fixed
    disable_all_gradients(model)

    # Train
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    if cuda: 
        model.cuda()
        probe.cuda()
    gpu_number = int(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda:0' if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
    lagrangian_optimization = LagrangianOptimization(optimizer,
                                                        device,
                                                        batch_size_multiplier=None)

    moving_average_window_size = 4
    f_moving_average = MovingAverage(window_size=moving_average_window_size)
    g_moving_average = MovingAverage(window_size=moving_average_window_size)

    analyse_n_epochs = [60, 50]
    for layer in range(2):
        print("Enabling layer "+str(layer))
        probe.enable_layer(layer)
        gnn = model.graph_net
        for e in range(analyse_n_epochs[layer]):
            for data in train_loader:
                loss, penalty, _, _, _, _, _ = train_or_eval_graphmask(probe, model, loss_function, data, gnn, cuda, train=True)

                g = torch.relu(loss - allowance).mean() * divergence_scaling
                f = penalty * penalty_scaling
                lagrangian_optimization.update(f, g)
                f_moving_average.register(float(f))
                g_moving_average.register(float(g))

                print(
                    "Running epoch {0:n} of GraphMask training. Mean divergence={1:.4f}, mean penalty={2:.4f}, loss={3:.4f}".format(
                        e,
                        g_moving_average.get_value(),
                        f_moving_average.get_value(),
                        loss.item()
                    )
                )

            # Valid & Test
            if (e+1) % test_every_n == 0:
                percent_div, sparsity = validate(probe, model, loss_function, valid_loader, gnn, cuda)
                print("Validate: percent_div: {0:.4f}. sparsity: {1:.4f}".format(percent_div, sparsity))

                if percent_div < max_allowed_performance_diff and sparsity <= best_sparsity:
                    print("Found better probe with sparsity={0:.4f}. Keeping these parameters.".format(sparsity))
                    best_sparsity = sparsity
                    best_path = './cache'+save_path+'_'+str(layer)+'_'+str(e)+'.pkl'
                    torch.save(probe.state_dict(), best_path)

                    percent_div, sparsity = validate(probe, model, loss_function, test_loader, gnn, cuda, split="test")
                    print("Test: percent_div: {0:.4f}. sparsity: {1:.4f}".format(percent_div, sparsity))
    
    probe.load_state_dict(torch.load(best_path))
    torch.save(probe.state_dict(), '.'+save_path+'.pkl')

    # Test
    percent_div, sparsity = validate(probe, model, loss_function, test_loader, gnn, cuda, split="test")
    print("Test: percent_div: {0:.4f}. sparsity: {1:.4f}".format(percent_div, sparsity))