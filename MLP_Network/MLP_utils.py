import os
import torch
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# Computer Accuracy
def compute_accuracy(model, data_loader, device):

    # Perform forward pass
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets  = targets.to(device)

            # Get the model outputs and compare them against labels
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            target_labels =  torch.argmax(targets, dim=1)
            correct_pred += (predicted_labels == target_labels).sum()

            # Compute accuracy
            accuracy = (correct_pred.float()/num_examples)*100

    return accuracy

# Plottings
def plot_training_loss(mini_batch_loss_list, num_epoch, iter_per_epoch,
                       result_dir=None, averaging_iteration=100):

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(mini_batch_loss_list)),
             mini_batch_loss_list, label='Minibatch Loss')

    if len(mini_batch_loss_list) < 1000:
        ax1.set_ylim([0, np.max(mini_batch_loss_list[1000:])*1.5])

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    ax1.plot(np.convolve(mini_batch_loss_list, np.ones(averaging_iteration,)/averaging_iteration,
                         mode='valid'),
             label='Running Average')

    ax1.legend()

    ax2 = ax1.twiny()
    newlabel = list(range(num_epoch+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())

    plt.tight_layout()


def plot_accuracy(train_acc_list, valid_acc_list, results_dir):
    num_epochs = len(train_acc_list)

    plt.plot(np.arange(1, num_epochs+1),
             train_acc_list, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc_list, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if results_dir is not None:
        img_path = os.path.join(results_dir, 'acc.pdf')
        plt.savefig(img_path)

def plot_confusion_matrix(conf_mat,
                          hide_spines=False,
                          hide_ticks=False,
                          figsize=None,
                          cmap=None,
                          colorbar=False,
                          show_absolute=True,
                          show_normed=False,
                          class_names=None):

    if not (show_absolute or show_normed):
        raise AssertionError('Both shows are false.')

    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError('len(class_names) should be equal to number of classes in the dataset.')

    total_samples   = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float')/total_samples

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)

    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat)*1.25, len(conf_mat)*1.25)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):

            cell_text = ""
            if show_absolute:
                cell_text += format(conf_mat[i, j], 'd')
                if show_normed:
                    cell_text += "\n" + '('
                    cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
                else:
                    cell_text += format(normed_conf_mat[i, j], '.2f')

                ax.text(x=j,
                        y=i,
                        s=cell_text,
                        va='center',
                        ha='center',
                        color="white" if normed_conf_mat[i,j] > 0.5 else "black")

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    return fig, ax

def compute_confusion_matrix(model, data_loader, device):
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets  = targets

            logits, _, _ = model(features)
            _, predicted_labels = torch.max(logits, 1)
            all_targets.extend(targets.to('cpu'))
            all_predictions.extend(predicted_labels.to('cpu'))

    all_predictions = all_predictions
    all_predictions = np.array(all_predictions)
    all_targets     = np.array(all_targets)

    class_labels = np.unique(np.concatenate((all_targets, all_predictions)))

    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])

    n_labels = class_labels.shape[0]

    lst = []
    z   = list(zip(all_targets, all_predictions))
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))

    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    return mat