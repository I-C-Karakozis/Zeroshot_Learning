import os
import sys
import copy
import time
import torch
import argparse
import numpy as np

from sklearn.neighbors import NearestNeighbors

from tools.loader    import get_data, get_dataloaders
from tools.log_utils import log_metrics, print_mode_info, print_border
from tools.models    import load_resnet18

# ---------------------------- Auxiliary Functions ---------------------------- #

def get_outputs(args, net, device, n_classes, dataloader):
    '''
    ASSUMPTION: The dataloader does not shuffle the examples.
    '''
    net.eval()
    all_y_scores = np.zeros((0, n_classes))

    with torch.no_grad():
        for inputs, _ in dataloader:            
            # collect network outputs matrix
            inputs = inputs.to(device)
            outputs = net(inputs).cpu().numpy()
            all_y_scores = np.concatenate((all_y_scores, outputs), axis=0)
            
            if args.debug: break

    return all_y_scores  

def get_good_attr(values, good_attr_id):
    return np.transpose(np.array([values[:, i] for i in good_attr_id]))

def measure_performance(dataset, neigh, inputs, labels, method):
    # compute accuracy
    preds = neigh.kneighbors(inputs, 1, return_distance=False).squeeze()
    correct = sum(np.equal(preds, labels))
    total = float(len(labels))
    acc = 100 * correct / total

    # log performance
    print('{} | Class Accuracy: {:.2f}% ({}/{})'.format(method, acc, correct, int(total)))
    sys.stdout.flush()

    # generate predictions txt
    with open('predictions/{}.txt'.format(method), 'w') as results:
        for i in range(len(preds)):
            img_path_split = dataset.img_paths[i].split('/')
            img = '{}/{}'.format(img_path_split[-2], img_path_split[-1])
            pred_class = dataset.classes[preds[i]]
            results.write('{} {}\n'.format(img, pred_class))

# ---------------------------- Model Training, Validation, and Evaluation ---------------------------- #

def __train(args, net, n_classes, criterion, optimizer, trainloader, device):
    net.train()
    train_loss = 0

    for inputs, targets in trainloader:
        inputs = inputs.to(device)
        targets = targets[:, :-1].squeeze(1).to(device)
        optimizer.zero_grad()

        # optimization
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if args.debug: break

    log_metrics(train_loss, len(trainloader))
    
def __validate(args, net, n_classes, criterion, devloader, device, cls_threshold=0.5):
    net.eval()
    dev_loss = 0
    total = 0.0
    correct = torch.zeros(n_classes).type(torch.FloatTensor).to(device)

    with torch.no_grad():
        for inputs, targets in devloader:
            inputs  = inputs.to(device)
            targets = targets[:, :-1].squeeze(1).to(device)
                
            # forward 
            outputs = net(inputs)
            dev_loss += criterion(outputs, targets).item()          

            # inference
            outputs = torch.sigmoid(outputs)
            preds = (outputs.data >= cls_threshold).type(torch.LongTensor).to(device)
            targets = targets.type(torch.LongTensor).to(device)
            correct += torch.sum(torch.eq(preds, targets), dim=0).type(torch.FloatTensor).to(device)
            total += targets.size(0)

            if args.debug: break

    log_metrics(dev_loss, len(devloader), correct, total, cls_threshold)
    return dev_loss / len(devloader)

def __test(dataset, attr_vectors, class_targets, outputs, preds, per_attr_acc, threshold):
    # get good attributes
    good_attr_id = []
    for attr_id, acc in enumerate(per_attr_acc):
        if acc > threshold: good_attr_id.append(attr_id)
    print('Evaluating @{:.2f} attribute accuracy using {} attributes.'.format(threshold, len(good_attr_id)))

    # get values corresponding to good attributes
    attr_vectors = get_good_attr(attr_vectors, good_attr_id)
    outputs = get_good_attr(outputs, good_attr_id)
    preds = get_good_attr(preds, good_attr_id)

    # setup nearest neighbors classification
    neigh_cosine = NearestNeighbors(n_neighbors=1, metric='cosine')
    neigh_cosine.fit(attr_vectors)
    neigh_l2 = NearestNeighbors(n_neighbors=1, metric='l2')
    neigh_l2.fit(attr_vectors)

    # measure class accuracy with good attributes
    measure_performance(dataset, neigh_cosine, outputs, class_targets, 'cosine_outputs')
    measure_performance(dataset, neigh_cosine, preds, class_targets, 'cosine_predictions')
    measure_performance(dataset, neigh_l2, outputs, class_targets, 'l2_outputs')
    measure_performance(dataset, neigh_l2, preds, class_targets, 'l2_predictions')

    print_border()

# ---------------------------- Exposed API ---------------------------- #

def train(args, ckpt, n_classes, net, device, trainloader, devloader):
    print_border(); print("TRAINING"); print_border()

    # setup best model validation
    backup_freq = 5
    best_loss = sys.maxsize
    best_val_epoch = 0
    best_net = None

    # loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    # optimization loop
    time_all = time.time()
    for epoch in range(0, args.epochs):
        print('Epoch: %d' % epoch)
        __train(args, net, n_classes, criterion, optimizer, trainloader, device)
        epoch_loss = __validate(args, net, n_classes, criterion, devloader, device)

        # update the best net
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_val_epoch = epoch
            best_net = copy.deepcopy(net.state_dict())

        if (epoch + 1) % backup_freq == 0: torch.save(net.state_dict(), 'models/'+ ckpt +"_last")
        if args.debug: break

    # timing
    time_elapsed = time.time() - time_all
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # save and load the best net
    print('Best Validation Loss: {:.4f} attained at epoch {}'.format(best_loss, best_val_epoch))
    best_net_filename = os.path.join('models', ckpt)
    torch.save(best_net, best_net_filename)
    net.load_state_dict(torch.load(best_net_filename))

    return net

def evaluate(args, net, device, dataset, dataloader):   
    # predict labels
    outputs = get_outputs(args, net, device, dataset.n_attributes, dataloader)
    outputs = torch.FloatTensor(outputs).to(device) 
    outputs = torch.sigmoid(outputs)
    preds = (outputs > 0.5).type(torch.LongTensor).to(device)

    # convert tensors to numpy
    outputs = outputs.cpu().numpy()
    preds = preds.cpu().numpy()
    targets = np.array(dataset.lbls)[:, :-1]
    if args.debug: targets = np.array(dataset.lbls)[:args.batch_size, :-1]

    # measure animal attribute accuracy
    correct = np.sum(np.equal(preds, targets), axis=0)
    total = len(preds)
    per_attr_acc = 100 * correct / total
    for attr_id, acc in enumerate(per_attr_acc):
        attr = dataset.id2attribute[attr_id]
        print("{}: {:.2f}%".format(attr, acc))
    print_border()

    # measure mean per class attribute accuracy
    print("Mean per Class Attribute Accuracy: ", np.mean(per_attr_acc))
    print_border()

    # get groundtruth values
    attr_vectors = np.array(dataset.get_attribute_vectors())
    class_targets = np.array(dataset.lbls)[:, -1].squeeze()
    if args.debug: class_targets = np.array(dataset.lbls)[:args.batch_size, -1].squeeze()

    # evaluate over multiple possible thresholds
    thresholds = [0] + [i for i in range(50, 95, 5)]
    for threshold in thresholds:
        __test(dataset, attr_vectors, class_targets, outputs, preds, per_attr_acc, threshold)

# ---------------------------- Core Pipeline ---------------------------- #

def pipeline(args):
    # load data
    if args.debug: args.batch_size = 2
    trainset, testset = get_data(args.datadir)
    trainloader, testloader = get_dataloaders(args.batch_size, trainset, testset)
    print_border()
    
    # load network
    n_classes = trainset.n_attributes
    net, device = load_resnet18(n_classes, pretrained=True)
    
    # log
    ckpt = "attr_mlc"
    if args.debug: ckpt += "_debug"
    print_mode_info(args, ckpt, device, n_classes)

    # optimization
    if args.model is None:
        net = train(args, ckpt, n_classes, net, device, trainloader, testloader)  
    else:
        net.load_state_dict(torch.load(os.path.join("models", args.model)))

    # evaluation
    print_border(); print("EVALUATION"); print_border()
    evaluate(args, net, device, testset, testloader)

# Python version: 3.6.7

if __name__ == '__main__':
    description = 'Train classifier on activity classification'
    parser = argparse.ArgumentParser(description)

    help_str = 'path to directory holding the images'
    parser.add_argument('--datadir', default='data/AwA_128x128', help=help_str)
    help_str = 'Filename of model to evaluate; leave None if you want to train model from scratch'
    parser.add_argument("--model", help=help_str)

    help_str = 'number of epochs'
    parser.add_argument('--epochs', default=20, type=int, help=help_str)
    help_str = 'batch size'
    parser.add_argument('--batch_size', default=64, type=int, help=help_str)

    help_str = 'learning rate'
    parser.add_argument('--lr', default=1e-5, type=float, help=help_str)
    help_str = 'weight decay'
    parser.add_argument('--weightdecay', default=1e-5, type=float, help=help_str)

    help_str = 'set to enable debug mode'
    parser.add_argument("--debug", action='store_true', default=False, help=help_str)

    ARGS = parser.parse_args()
    pipeline(ARGS)
