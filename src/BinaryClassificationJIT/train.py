import datetime
import os
import sys
import time
import signal
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.backends import mps

from torchvision import datasets, transforms

# define hyperparameters

LEARNING_RATE = 0.001
MINI_BATCH_SIZE = 500
MINI_BATCH_LOG_INTERVAL = 10
TRAINING_EPOCHS = 2
# MOMENTUM = None

# load training and test data

INPUT_DIM = 28 # 28x28 image

TRAIN_DATASET = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize((INPUT_DIM, INPUT_DIM)),
        transforms.ToTensor()
    ])
)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=MINI_BATCH_SIZE, shuffle=False)

VALIDATION_DATASET = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.Resize((INPUT_DIM, INPUT_DIM)),
        transforms.ToTensor()
    ])
)
VALIDATION_DATALOADER = DataLoader(VALIDATION_DATASET, batch_size=MINI_BATCH_SIZE, shuffle=False)

TEST_DATASET = None
TEST_DATALOADER = None

# Load saved model if exist for when resuming training

BEST_VALIDATION_MODEL_SAVE_PATH = os.path.join('checkpoint_v_mcc_model.pth')
BEST_TESTING_MODEL_SAVE_PATH    = os.path.join('checkpoint_t_mcc_model.pth')

# just delete or rename the model if you want to start training again
if os.path.isfile(BEST_TESTING_MODEL_SAVE_PATH):
    print('Training Checkpoint Found: Resuming Training')
    MODEL = torch.jit.load(BEST_TESTING_MODEL_SAVE_PATH)
elif os.path.isfile(BEST_VALIDATION_MODEL_SAVE_PATH):
    print('Validation Checkpoint Found: Resuming Training')
    MODEL = torch.jit.load(BEST_VALIDATION_MODEL_SAVE_PATH)
else:
    print('Error: jit\'ed model not found!')
    exit(1)

# get best available compute device and load the model to it

DEVICE = (
         "cuda" if torch.cuda.is_available()
    else "mps"  if mps.is_available()
    else "cpu"
)

MODEL.to(DEVICE)
MODEL.eval()
print(f'Device Loaded To : {DEVICE}')

# Simple Input Test

initial_test_output = MODEL(torch.ones(1, 1, 28, 28))
print('initial_test_output :\n', initial_test_output, '\n')
print('initial_test_output :\n', initial_test_output.argmax(1), '\n\n')

# Initialize optimizer

OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

# initialize loss/cost function

# Common loss functions include nn.MSELoss (Mean Square Error) for
# regression tasks, and nn.NLLLoss (Negative Log Likelihood) for
# classification. nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.

LOSS_FUNCTION = torch.nn.CrossEntropyLoss()

# register eventhandler to save model on exit


# defined test loop

def test_loop(dataset, dataloader):
    
    # Set the MODEL to evaluation mode - important for batch normalization and dropout layers
    
    MODEL.eval()

    total_dataset_size = len(dataset)
    total_mini_batches = len(dataloader)
    total_samples_done = 0
    average_loss, correct = 0, 0

    # Evaluating the MODEL with torch.no_grad() ensures that no gradients
    # are computed during test mode also serves to reduce unnecessary gradient
    # computations and memory usage for tensors with requires_grad=True

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:

            batch_inputs.to(DEVICE)
            batch_labels.to(DEVICE)

            batch_output = MODEL(batch_inputs)

            average_loss += LOSS_FUNCTION(batch_output, batch_labels).item()
            correct += (batch_output.argmax(1) == batch_labels).type(torch.float).sum().item()
            total_samples_done += len(batch_labels)

    # calculate average loss and the accuracy of model during test

    average_loss /= total_mini_batches
    accuracy = correct / total_dataset_size


    print(f"test_loop  | Accuracy({int(correct)}/{int(total_samples_done)}): {100 * accuracy:>0.2f}%, Avg. Loss: {average_loss:>8f} \n")

    return accuracy, average_loss

# main loop

if __name__ == "__main__":
    LAST_EPOCH = 0

    print(f'Hyperparameters: \nLearning Rate = {LEARNING_RATE}\nMini-Batch Size = {MINI_BATCH_SIZE}\nEpochs = {LAST_EPOCH + 1}/{TRAINING_EPOCHS}')

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/training_{}'.format(timestamp))

    # # write model graph in pytorch
    # writer.add_graph(MODEL, torch.rand(1, 1, INPUT_DIM, INPUT_DIM))

    best_validation_loss = 1_000_000.
    best_test_loss = 1_000_000.0

    # dataset info

    total_training_mini_batches = len(TRAIN_DATALOADER)
    total_training_data = len(TRAIN_DATASET)

    # per epoch training

    start_time = time.time()
    epoch_idx = 0
    
    while epoch_idx < TRAINING_EPOCHS:

        # Make sure gradient tracking is on, and do a pass over the data
        MODEL.train(True)

        # =========== TRAINING LOOP STARTS HERE =============
        
        training_amassed_loss = 0.
        training_correct_pred = 0.

        training_overall_samples_done = 0.
        training_current_log_samples_done = 0.
        training_log_counts = 0.

        for training_mini_batch_idx, training_mini_batch in enumerate(TRAIN_DATALOADER):

            # load the inputs and labels from the dataloader

            batch_inputs = training_mini_batch[0].to(DEVICE)
            batch_labels = training_mini_batch[1].to(DEVICE)

            # Set the MODEL to training mode - important for batch normalization and dropout layers

            MODEL.to(DEVICE)
            MODEL.train()

            # zero your gradients to prevent gradient accumulation

            OPTIMIZER.zero_grad()

            # feed the inputs to the network

            batch_output: torch.Tensor = MODEL(batch_inputs)

            # calculate the loss of the network

            loss = LOSS_FUNCTION(batch_output, batch_labels)

            # calculate the gradients of the network and perform backpropagation

            loss.backward()
            OPTIMIZER.step()

            # get the loss scalar value and calculate the number of processed data from the dataset

            loss, samples_processed = loss.item(), training_mini_batch_idx * MINI_BATCH_SIZE + len(batch_inputs)

            # accumulate loss in each mini-batches

            training_amassed_loss += loss
            training_correct_pred += (batch_output.argmax(1) == batch_labels).type(torch.float).sum().item()
            training_overall_samples_done += len(batch_inputs)
            training_current_log_samples_done += len(batch_inputs)
            training_log_counts += 1

            # log loss every `minibatch_log_interval`
            if (not ((training_mini_batch_idx + 1) % MINI_BATCH_LOG_INTERVAL)) or training_mini_batch_idx == 0:

                # calculate the training accuracy and loss for each minibatch log interval

                training_running_loss = training_amassed_loss / training_log_counts
                training_running_accuracy = training_correct_pred / training_current_log_samples_done

                print(f"train_loop | Epoch : {epoch_idx + 1}/{TRAINING_EPOCHS} | Accuracy({int(training_correct_pred)}/{int(training_current_log_samples_done)}): {training_running_accuracy * 100:>0.2f}% | Loss: {loss:>7f} | [{samples_processed:>5d}/{total_training_data:>5d}]")

                # Write Training Reports to Tensorboard

                writer.add_scalar(
                    f'Accuracy every {MINI_BATCH_LOG_INTERVAL} mini-batches: Training',
                    training_running_accuracy,
                    epoch_idx * total_training_mini_batches + training_mini_batch_idx
                )

                writer.add_scalar(
                    f'Loss every {MINI_BATCH_LOG_INTERVAL} mini-batches: Training',
                    training_running_loss,
                    epoch_idx * total_training_mini_batches + training_mini_batch_idx
                )

                # if there is a test dataloader perform validation runs and write reports to tensorboard

                if VALIDATION_DATASET != None and VALIDATION_DATALOADER != None:

                    #========== run validation every after training ==========#

                    validation_accuracy, validation_loss = test_loop(VALIDATION_DATASET, VALIDATION_DATALOADER)
                    
                    # Set the MODEL back to training mode after tests
                    
                    MODEL.train()

                    # write specific reports to tensorboard

                    writer.add_scalar(
                        f'Accuracy every {MINI_BATCH_LOG_INTERVAL} mini-batches: Validation',
                        validation_accuracy,
                        epoch_idx * total_training_mini_batches + training_mini_batch_idx
                    )

                    writer.add_scalars(
                        f'Accuracy every {MINI_BATCH_LOG_INTERVAL} mini-batches: Training vs Validation',
                        {
                            'Training': training_running_accuracy,
                            'Validation': validation_accuracy
                        },
                        epoch_idx * total_training_mini_batches + training_mini_batch_idx
                    )

                    writer.add_scalars(
                        f'Avg. Loss per {MINI_BATCH_LOG_INTERVAL} mini-batches: Training vs Validation',
                        {
                            'Training': training_running_loss,
                            'Validation': validation_loss
                        },
                        epoch_idx * total_training_mini_batches + training_mini_batch_idx
                    )

                    # Track best performance, and save the model's state
                    if validation_loss < best_validation_loss:

                        best_validation_loss = validation_loss

                        try:
                            MODEL.eval()
                            MODEL.to("cpu")
                            torch.jit.save(MODEL, BEST_VALIDATION_MODEL_SAVE_PATH)
                            print('New Best Validation Result Model Saved')
                        except Exception as e:
                            print(f'Error Occured: {e}')

                # zero out accumulated training data to get the next proper running averages

                training_amassed_loss = 0.0
                training_correct_pred = 0.0

                training_log_counts = 0
                training_current_log_samples_done = 0.0

        #========== TRAINING LOOP ENDS HERE ==========#

        #========== RUN TEST ON TEST DATASET EVERY EPOCH ==========#

        if TEST_DATASET != None and TEST_DATALOADER != None:
            test_accuracy, test_loss = test_loop(TEST_DATASET, TEST_DATALOADER)

            print(f'\nEpoch Test Accuracy : {test_accuracy}')
            print(f'Epoch Test Loss       : {test_loss}')

            writer.add_scalar(
                f'Test Accuracy Every Epoch',
                test_accuracy,
                epoch_idx
            )

            writer.add_scalar(
                f'Test Loss Every Epoch',
                test_loss,
                epoch_idx
            )

            if test_loss < best_test_loss:
                best_test_loss = test_loss

                try:
                    MODEL.eval()
                    MODEL.to("cpu")
                    torch.jit.save(MODEL, BEST_TESTING_MODEL_SAVE_PATH)
                    print('New Best Test Result Model Saved')
                except Exception as e:
                    print(f'Error Occured: {e}')

        writer.flush()
        epoch_idx += 1
        LAST_EPOCH = epoch_idx

    #========== END OF TRAINING EPOCH LOOP ==========#

    writer.close()
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
