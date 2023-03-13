import time
from MLP_utils import *
from MLP_data import *
import torch.optim as opt
from MLP_model import MLP

# Train the MLP model
def train_model(model, num_epochs,
                train_loader, valid_loader, test_loader,
                optimizer, device, logging_interval=50,
                scheduler=None, scheduler_on='valid_acc'):

    # Logging time
    start_time = time.time()
    mini_batch_loss_list, train_acc_list, valid_acc_list = [], [], []

    # Run through epochs
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            # Forward pass
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)

            # Computer loss
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            # Backprop
            loss.backward()
            optimizer.step()

            # Gather mini batch loss and log
            mini_batch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d}'
                      f' | Batch {batch_idx:04d}/{len(train_loader):04d}'
                      f' | Loss: {loss:.4f}')

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc = compute_accuracy(model, valid_loader, device=device)
            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d}'
                  f' | Train: {train_acc:.2f}%'
                  f' | Validation: {valid_acc:.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')

        if scheduler is not None:
            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(mini_batch_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy: {test_acc:.2f}%')

    return mini_batch_loss_list, train_acc_list, valid_acc_list


def main():

    # Data file and word embedding model
    data_file_path = './train.txt'
    sentences, pos_to_idx = get_sentences_and_tags(data_file_path)
    glove_model = load_glove_model()

    # Device initialization
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: ", DEVICE)

    # Settings and initialization
    BATCH_SIZE = 32
    test_fraction, val_fraction  = 0.1, 0.1
    train_loader, valid_loader, test_loader = get_dataloaders(
        glove_model = glove_model,
        sentences   = sentences,
        pos_to_idx  = pos_to_idx,
        batch_size  = BATCH_SIZE,
        val_frac    = val_fraction,
        test_frac   = test_fraction)

    for (feature, label) in train_loader:
        feature_length = feature.shape[1]
        break

    # Initialize model and model parameters
    nEpochs = 30
    learning_rate = 0.01

    input_layer_size    = feature_length
    hidden_layer_1_size = 1024
    hidden_layer_2_size = 4096
    hidden_layer_3_size = 2048
    ouput_layer_size    = len(pos_to_idx)

    model = MLP(input_layer_size=input_layer_size,
                hidden_layer_1_size=hidden_layer_1_size,
                hidden_layer_2_size=hidden_layer_2_size,
                hidden_layer_3_size=hidden_layer_3_size,
                output_layer_size=ouput_layer_size)

    model = model.to(DEVICE)
    optimizer = opt.SGD(model.parameters(), momentum=0.9, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.1,
                                                           mode='max',
                                                           verbose=True)

    mini_batch_loss_list, train_acc_list, valid_acc_list = train_model(
        model            = model,
        num_epochs       = nEpochs,
        train_loader     = train_loader,
        valid_loader     = valid_loader,
        test_loader      = test_loader,
        optimizer        = optimizer,
        device           = DEVICE,
        logging_interval = 500,
        scheduler        = scheduler,
        scheduler_on     = 'valid_acc'
    )

    os.makedirs('../save_model', exist_ok=True)
    torch.save(model.state_dict(), './save_model/MLP.pt')
    print(f"Model saved.")

    plot_training_loss(mini_batch_loss_list=mini_batch_loss_list,
                       num_epoch=nEpochs,
                       iter_per_epoch=len(train_loader),
                       result_dir=None,
                       averaging_iteration=200
                       )

    plt.show()

    plot_accuracy(train_acc_list=train_acc_list,
                  valid_acc_list=valid_acc_list,
                  results_dir=None
    )
    plt.ylim([10, 100])
    plt.show()

    # mat = compute_confusion_matrix(model=model, data_loader=test_loader, device=torch.device('cpu'))
    # plot_confusion_matrix(mat, figsize=(10, 10), show_absolute=True)
    # plt.show()

if __name__=='__main__':
    main()


