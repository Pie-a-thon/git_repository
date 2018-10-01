from matplotlib import pyplot as plt
import pickle
import numpy as np

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def acc_losses():
    losses0 = load_obj("losses_cells")
    losses_no_cells0 = load_obj("losses_no_cells")
    losses1 = load_obj("losses_cells_1")
    losses_no_cells1 = load_obj("losses_no_cells_1")
    losses_squares = load_obj("losses_squares")

    acc0 = load_obj("accuracy_cells")
    acc_no_cells0 = load_obj("accuracy_no_cells")
    acc1 = load_obj("accuracy_cells_1")
    acc_no_cells1 = load_obj("accuracy_no_cells_1")
    acc_squares = load_obj("accuracy_squares")


    batches0 = np.linspace(511.0, len(losses0)-1, num = 24, dtype=int)
    batches1 = np.linspace(381.0, len(losses1)-1, num = 24, dtype=int)
    batches = np.hstack([batches0, batches1])
    losses = np.hstack([losses0[batches0], losses1[batches1]])
    losses_no_cells = np.hstack([losses_no_cells0[batches0], losses_no_cells1[batches1]])
    acc = np.hstack([acc0[batches0], acc1[batches1]])
    acc_no_cells = np.hstack([acc_no_cells0[batches0], acc_no_cells1[batches1]])
    epochs0 = np.linspace(0, 24, num=24)
    epochs = np.linspace(0, 48, num=48)

    plt.plot(epochs, losses)
    plt.plot(epochs, losses_no_cells)
    plt.plot(epochs0, losses_squares[batches0])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['With cells', 'Without cells', "with squares"], loc='upper left')

    plt.figure(2)
    plt.plot(epochs, acc)
    plt.plot(epochs, acc_no_cells)
    plt.plot(epochs0, acc_squares[batches0])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['With cells', 'Without cells', "with squares"], loc='upper left')
    plt.show()

losses0 = load_obj("losses_ttbar_1")
losses1 = load_obj("losses_ttbar_2")

acc0 = load_obj("accuracy_ttbar_1")
acc1 = load_obj("accuracy_ttbar_2")

print(len(acc1)/24)
batches0 = np.linspace(12.0, len(losses0)-1, num = 24, dtype=int)
batches1 = np.linspace(26.0, len(losses1)-1, num = 24, dtype=int)
batches = np.hstack([batches0, batches1])
losses = np.hstack([losses0[batches0], losses1[batches1]])
acc = np.hstack([acc0[batches0], acc1[batches1]])
epochs0 = np.linspace(0, 24, num=24)
epochs = np.linspace(0, 48, num=48)

plt.plot(epochs, losses)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['With cells', 'Without cells', "with squares"], loc='upper left')

plt.figure(2)
plt.plot(epochs, acc)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['With cells', 'Without cells', "with squares"], loc='upper left')
plt.show()
