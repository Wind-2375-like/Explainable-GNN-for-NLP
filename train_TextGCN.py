import torch
import random
import argparse
import numpy as np
from time import time
from model import TextGCN
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from graphmask_probe import GraphMaskProbe
from tools.utils.moving_average import MovingAverage
from tools.utils.torch_utils.lagrangian_optimization import LagrangianOptimization

from utils import get_time_dif, LoadData

parser = argparse.ArgumentParser(description='TextGCN')
parser.add_argument('--model', type=str, default='TextGCN', help='choose a model')
args = parser.parse_args()
# seed = random.randint(0, 100000)
seed = 11119


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TextGCNTrainer:
    def __init__(self, args, model, pre_data):
        self.args = args
        self.model = model
        self.device = args.device
        self.max_epoch = self.args.max_epoch
        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = EarlyStopping(args.early_stopping)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_path = "./TextGCN_datasets/model/model.pkl"

    def fit(self):
        self.prepare_data()
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        print(self.model.parameters)
        self.model_param = sum(param.numel() for param in self.model.parameters())
        print('model parameters:', self.model_param)
        self.convert_tensor()
        self.train()
        self.test()
        return self.model

    def prepare_data(self):
        self.target = self.predata.target
        self.nclass = self.predata.nclass
        self.data = self.predata.graph

        self.train_lst, self.val_lst = train_test_split(self.predata.train_lst,
                                                        test_size=self.args.val_ratio,
                                                        shuffle=True,
                                                        random_state=self.args.seed)
        self.test_lst = self.predata.test_lst

    def convert_tensor(self):
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.target = torch.tensor(self.target).long().to(self.device)
        self.train_lst = torch.tensor(self.train_lst).long().to(self.device)
        self.val_lst = torch.tensor(self.val_lst).long().to(self.device)

    def train(self):
        start_time = time()
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model.forward(self.data)
            loss = self.criterion(logits[self.train_lst],
                                  self.target[self.train_lst])

            loss.backward()
            self.optimizer.step()
            pred = torch.max(logits[self.train_lst].data, 1)[1].cpu().numpy()
            target = self.target[self.train_lst].data.cpu().numpy()
            train_acc = accuracy_score(pred, target)
            val_loss, val_acc, val_f1 = self.val(self.val_lst)
            time_dif = get_time_dif(start_time)
            msg = 'Epoch: {:>2},  Train Loss: {:>6.3}, Train Acc: {:>6.2%}, Val Loss: {:>6.3},  Val Acc: {:>6.2%},  Time: {}'
            print(msg.format(epoch, loss.item(), train_acc, val_loss, val_acc, time_dif))
            if self.earlystopping(val_loss):
                print("...")
                break


    @torch.no_grad()
    def val(self, x, test=False):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(self.data)
            loss = self.criterion(logits[x],
                                  self.target[x])

            pred = torch.max(logits[x].data, 1)[1].cpu().numpy()
            target = self.target[x].data.cpu().numpy()
            acc = accuracy_score(pred, target)
            f1 = f1_score(pred, target, average='macro')
        if test:
            report = metrics.classification_report(pred, target, digits=4)
            # report = metrics.classification_report(pred, target, target_names=config.class_list, digits=4)
            confusion = metrics.confusion_matrix(pred, target)
            return acc, report, confusion
        return loss.item(), acc, f1

    @torch.no_grad()
    def test(self):
        self.test_lst = torch.tensor(self.test_lst).long().to(self.device)
        acc, report, confusion = self.val(self.test_lst, test=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'acc': acc,
            'report': report,
            'confusion': confusion,
            'predata': self.predata,
        }, self.model_path)

        msg = '\nTest Acc: {:>6.2%}'
        print(msg.format(acc))
        print("Precision, Recall and F1-Score...")
        print(report)
        print("Confusion Matrix...")
        print(confusion)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_path = "./TextGCN_datasets/model/model.pkl"

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_path)
        self.val_loss_min = val_loss

    def load_model(self):
        return torch.load(self.model_path)


class TextGCNAnalyser:
    def __init__(self, model, args, pre_data):
        self.best_sparsity = 1.01
        self.allowance = 0.03
        self.divergence_scaling = 1
        self.penalty_scaling = 5
        self.max_allowed_performance_diff = 0.05
        self.save_path = "textgcn_model_probe"
        self.best_path = None
        self.learning_rate = 3e-4
        self.cuda = True
        self.moving_average_window_size = 4
        self.layer_num = 2
        self.analyse_n_epochs = [1800, 1800]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = model
        self.args = args
        self.predata = pre_data
        self.framework = TextGCNTrainer(args=self.args, model=self.model, pre_data=self.predata)


    def initialise_for_model(self, model):
        vertex_embedding_dims = np.array([model.conv1.out_channels, model.conv2.out_channels])
        message_dims = np.array([model.conv1.out_channels, model.conv2.out_channels])
        probe = GraphMaskProbe(vertex_embedding_dims, message_dims, message_dims)
        return probe


    def disable_all_gradients(self, module):
        for param in module.parameters():
            param.requires_grad = False


    def train_or_eval_graphmask(self, probe, model, loss_function, data_lst, trainer, cuda, train=False):
        model.eval()
        logits = model.forward(trainer.data)
        loss_ori = loss_function(logits[data_lst],
                                trainer.target[data_lst])
        pred_ori = torch.max(logits[data_lst], 1)[1].cpu().numpy()
        label = trainer.target[data_lst].data.cpu().numpy()

        gates, baselines, penalty = probe(model)
        model.inject_message_scale(gates)
        model.inject_message_replacement(baselines)
        
        logits = model.forward(trainer.data)
        loss_aft = loss_function(logits[data_lst],
                                trainer.target[data_lst])
        pred_aft = torch.max(logits[data_lst], 1)[1].cpu().numpy()

        return torch.abs(loss_aft-loss_ori), penalty, gates, pred_ori, pred_aft, label


    def validate(self, probe, model, loss_function, data_lst, trainer, cuda, split="dev"):
        all_gates = 0
        all_messages = 0
        layer0_gates = 0
        layer0_messages = 0
        layer1_gates = 0
        layer1_messages = 0

        with torch.no_grad():
            _, _, gates, pred_ori, pred_aft, label = self.train_or_eval_graphmask(probe, model, loss_function, data_lst, trainer, cuda)
            
            all_gates += float(sum([g.sum().detach().cpu() for g in gates]))
            all_messages += float(model.count_latest_messages())
            layer0_gates += float(gates[0].sum().detach().cpu())
            layer1_gates += float(gates[1].sum().detach().cpu())
            layer0_messages += float(model.get_latest_messages()[0].numel()/model.get_latest_messages()[0].shape[-1])
            layer1_messages += float(model.get_latest_messages()[1].numel()/model.get_latest_messages()[1].shape[-1])

            acc_ori = accuracy_score(pred_ori, label)
            acc_aft = accuracy_score(pred_aft, label)
            f1_ori  = f1_score(pred_ori, label, average='macro')
            f1_aft  = f1_score(pred_aft, label, average='macro')

            print("GraphMask comparison on the "+split+"-split:")
            print("======================================")
            print("Original accuracy: {0:.4f}, fscore: {1:.4f}".format(acc_ori, f1_ori))
            print("Gated accuracy: {0:.4f}, fscore: {1:.4f}".format(acc_aft, f1_aft))
            print("Retained messages: all: {0:.4f}, layer0: {1:.4f}, layer1: {2:.4f}".format(
                all_gates / all_messages, 
                layer0_gates / layer0_messages,
                layer1_gates / layer1_messages
            ))

            report, confusion = None, None
            if split == 'test':
                report = metrics.classification_report(pred_aft, label, digits=4)
                confusion = metrics.confusion_matrix(pred_aft, label)
            
            sparsity = float(all_gates / all_messages)

            diff_f1 = np.abs(f1_ori - f1_aft)
            diff_acc = np.abs(acc_ori - acc_aft)
            percent_div = np.mean([diff_f1 / (f1_ori + 1e-8), diff_acc / (acc_ori + 1e-8)])

            return percent_div, sparsity, report, confusion


    def train(self, model, probe):
        trainer = self.framework
        trainer.prepare_data()
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.learning_rate)
        
        device = torch.device('cuda:3' if self.cuda else 'cpu')
        if self.cuda: 
            model.to(device)
            probe.to(device)

        trainer.convert_tensor()
        lagrangian_optimization = LagrangianOptimization(optimizer,
                                                            device,
                                                            batch_size_multiplier=None)
        f_moving_average = MovingAverage(window_size=self.moving_average_window_size)
        g_moving_average = MovingAverage(window_size=self.moving_average_window_size)
        loss_function = self.criterion
        train_lst = trainer.train_lst
        val_lst = trainer.val_lst
        test_lst = trainer.test_lst

        for layer in range(self.layer_num):
            print("Enabling layer "+str(layer))
            probe.enable_layer(layer)
            for e in range(self.analyse_n_epochs[layer]):
                loss, penalty, _, _, _, _ = self.train_or_eval_graphmask(probe, model, loss_function, train_lst, trainer, self.cuda, train=True)
                g = torch.relu(loss - self.allowance).mean() * self.divergence_scaling
                f = penalty * self.penalty_scaling
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

                # Validate
                percent_div, sparsity, _, _ = self.validate(probe, model, loss_function, val_lst, trainer, self.cuda)
                print("Validate: percent_div: {0:.4f}. sparsity: {1:.4f}".format(percent_div, sparsity))
                if percent_div < self.max_allowed_performance_diff and sparsity <= self.best_sparsity:
                    print("Found better probe with sparsity={0:.4f}. Keeping these parameters.".format(sparsity))
                    self.best_sparsity = sparsity
                    self.best_path = './cache/'+self.save_path+'_'+str(layer)+'_'+str(e)+'.pkl'
                    torch.save(probe.state_dict(), self.best_path)

                    percent_div, sparsity, _, _ = self.validate(probe, model, loss_function, test_lst, trainer, self.cuda, split="test")
                    print("Test: percent_div: {0:.4f}. sparsity: {1:.4f}".format(percent_div, sparsity))

        probe.load_state_dict(torch.load(self.best_path))
        torch.save(probe.state_dict(), self.save_path+'.pkl')

        # Test
        percent_div, sparsity, report, confusion = self.validate(probe, model, loss_function, test_lst, trainer, self.cuda, split="test")
        print("Test: percent_div: {0:.4f}. sparsity: {1:.4f}".format(percent_div, sparsity))
        print("Precision, Recall and F1-Score...")
        print(report)
        print("Confusion Matrix...")
        print(confusion)


    def fit(self, model):
        # Initialising
        probe = self.initialise_for_model(model)

        # Make pretrained model fixed
        self.disable_all_gradients(model)

        # Train
        self.train(model, probe)

def print_msg(acc, report, confusion):
    msg = '\nTest Acc: {:>6.2%}'
    print(msg.format(acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)


def run(dataset, seed=seed):
    args.dataset = dataset
    args.device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
    args.nhid = 100
    args.max_epoch = 200
    args.dropout = 0.5
    args.val_ratio = 0.1
    args.early_stopping = 10
    args.lr = 0.02
    args.seed = seed
    args.load = True
    # args.load = False
    print(args)

    seed_everything()
    predata = LoadData(args)
    model = TextGCN(nfeat=predata.nfeat_dim,
                nhid=args.nhid,
                nclass=predata.nclass,
                dropout=args.dropout)

    if args.load == False:
        framework = TextGCNTrainer(model=model, args=args, pre_data=predata)
        model = framework.fit()
    else:
        checkpoint = torch.load('./TextGCN_datasets/model/model.pkl')
        model.load_state_dict(checkpoint['model_state_dict'])
        acc, report, confusion, predata = checkpoint['acc'], checkpoint['report'], checkpoint['confusion'], checkpoint['predata']
        print_msg(acc, report, confusion)

    # Analysing
    print("=============Analysing...=============")
    analyser = TextGCNAnalyser(model=model, args=args, pre_data=predata)
    analyser.fit(model)

if __name__ == '__main__':
    # run("mr")
    run("R8")