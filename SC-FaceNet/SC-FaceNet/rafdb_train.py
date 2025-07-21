import os
import sys
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import itertools
import torch.nn.functional as F
from networks.BAM import SCFaceNet
from sklearn.metrics import confusion_matrix
from sam import SAM
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

eps = sys.float_info.epsilon


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default=r'', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.00001, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=12, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=60, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    return parser.parse_args()


class RFLoss(nn.Module):
    def __init__(self, ):
        super(RFLoss, self).__init__()

    def forward(self, x):
        BAUs = len(x)
        loss = 0
        cnt = 0
        if BAUs > 1:
            for i in range(BAUs - 1):
                for j in range(i + 1, BAUs):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt + 1
                    loss = loss + mse
            loss = cnt / (loss + eps)
        else:
            loss = 0
        return loss


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          fontsize=10):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=10)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=8,fontweight='bold')
    plt.yticks(tick_marks, classes, rotation=45,fontsize=8,fontweight='bold')

    fmt = '.1f' if normalize else 'd'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=fontsize,fontweight='bold')

    plt.ylabel('Actual', fontsize=10,fontweight='bold')
    plt.xlabel('Predicted', fontsize=10,fontweight='bold')
    plt.tight_layout()


class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']


def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = SCFaceNet(num_class=7, num_head=args.num_head)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(5),
            transforms.RandomCrop(112, padding=8)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    train_dataset = datasets.ImageFolder(f'{args.raf_path}/train', transform=data_transforms)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    val_dataset = datasets.ImageFolder(f'{args.raf_path}/val', transform=data_transforms_val)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    criterion_cls = torch.nn.CrossEntropyLoss()

    criterion_rf = RFLoss()
    optimizer = SAM(model.parameters(), torch.optim.Adam, lr=args.lr, rho=0.05, adaptive=False, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)

            out, feat, BAUs = model(imgs)
            loss = (criterion_cls(out, targets) + 0.5* criterion_rf(BAUs))

            loss.backward()
            optimizer.first_step(zero_grad=True)

            imgs = imgs.to(device)
            targets = targets.to(device)

            out, feat, BAUs = model(imgs)

            loss = criterion_cls(out, targets) + 0.5* criterion_rf(BAUs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (
        epoch, acc, running_loss, optimizer.param_groups[0]['lr']))

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0

            y_true = []
            y_pred = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                out, feat, BAUs = model(imgs)
                loss = criterion_cls(out, targets) + 0.5* criterion_rf(BAUs)

                running_loss += loss

                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += imgs.size(0)

                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())

                if iter_cnt == 0:
                    all_predicted = predicts
                    all_targets = targets
                else:
                    all_predicted = torch.cat((all_predicted, predicts), 0)
                    all_targets = torch.cat((all_targets, targets), 0)
                iter_cnt += 1
            running_loss = running_loss / iter_cnt
            scheduler.step()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            best_acc = max(acc, best_acc)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')

            tqdm.write(
                "[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f. Precision: %.4f. Recall: %.4f. F1: %.4f" % (epoch, acc, balanced_acc, running_loss, precision, recall, f1))
            tqdm.write("best_acc:" + str(best_acc))

            if acc > 0.85 and acc == best_acc:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('checkpoints', 'checkpoints_ver3',
                                        "rafdb_epoch" + str(epoch) + "_acc" + str(acc) + "_bacc" + str(
                                            balanced_acc) + ".pth"))
                tqdm.write('Model saved.')


                matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
                np.set_printoptions(precision=2)
                plt.figure(figsize=(7, 5))
                plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                                      title='RAF-DB Confusion Matrix (acc: %0.2f%%)' % (acc * 100),
                                      fontsize=10)

                plt.savefig(os.path.join('checkpoints', 'checkpoints_ver3',
                                         "rafdb_epoch" + str(epoch) + "_acc" + str(acc) + "_bacc" + str(
                                             balanced_acc) + ".png"))
                plt.close()


if __name__ == "__main__":
    run_training()
