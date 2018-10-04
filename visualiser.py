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

def ttbar():
    losses00 = load_obj("losses_ttbar_1")
    losses01 = load_obj("losses_ttbar_2")
    losses02 = load_obj("losses_ttbar_3")


    losses10 = load_obj("losses_ttbar_1_squares")
    losses11 = load_obj("losses_ttbar_2_squares")
    losses12 = load_obj("losses_ttbar_3_squares")


    losses20 = load_obj("losses_ttbar_1_mapped")
    losses21 = load_obj("losses_ttbar_2_mapped")
    losses22 = load_obj("losses_ttbar_3_mapped")


    acc00 = load_obj("accuracy_ttbar_1")
    acc01 = load_obj("accuracy_ttbar_2")
    acc02 = load_obj("accuracy_ttbar_3")


    acc10 = load_obj("accuracy_ttbar_1_squares")
    acc11 = load_obj("accuracy_ttbar_2_squares")
    acc12 = load_obj("accuracy_ttbar_3_squares")


    acc20 = load_obj("accuracy_ttbar_1_mapped")
    acc21 = load_obj("accuracy_ttbar_2_mapped")
    acc22 = load_obj("accuracy_ttbar_3_mapped")

    batches00 = np.linspace(len(acc00)/24, len(losses00)-1, num = 24, dtype=int)
    batches01 = np.linspace(len(acc01)/24, len(losses01)-1, num = 24, dtype=int)
    batches02 = np.linspace(len(acc02)/24, len(losses02)-1, num = 24, dtype=int)


    losses = np.hstack([losses00[batches00], losses01[batches01], losses02[batches02]])
    acc = np.hstack([acc00[batches00], acc01[batches01], acc02[batches02]])
    losses_sq = np.hstack([losses10[batches00], losses11[batches01], losses12[batches02]])
    acc_sq = np.hstack([acc10[batches00], acc11[batches01], acc12[batches02]])
    losses_mapped = np.hstack([losses20[batches00], losses21[batches01], losses22[batches02]])
    acc_mapped = np.hstack([acc20[batches00], acc21[batches01], acc22[batches02]])

    epochs0 = np.linspace(0, 72, num=72)
    epochs1 = np.linspace(0, 48, num=48)

    plt.plot(epochs0, losses)
    plt.plot(epochs0, losses_sq)
    plt.plot(epochs0, losses_mapped)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['xyz only', "with squares", "with mapped"], loc='upper left')

    plt.figure(2)
    plt.plot(epochs0, acc)
    plt.plot(epochs0, acc_sq)
    plt.plot(epochs0, acc_mapped)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['xyz only', "with squares", "with mapped"], loc='upper left')
    plt.show()

acc=load_obj("accuracy_ttbar")
acc_mapped = load_obj("accuracy_ttbar_mapped")
acc_sq=load_obj("accuracy_ttbar_squares")

losses=load_obj("losses_ttbar")
losses_mapped=load_obj("losses_ttbar_mapped")
losses_sq=load_obj("losses_ttbar_squares")

batches = np.linspace(len(acc)/24, len(losses)-1, num = 24, dtype=int)
epochs = np.linspace(0, 24, num=24)

plt.plot(epochs, losses[batches])
plt.plot(epochs, losses_sq[batches])
plt.plot(epochs, losses_mapped[batches])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['xyz only', "with squares", "with mapped"], loc='upper left')

plt.figure(2)
plt.plot(epochs, acc[batches])
plt.plot(epochs, acc_sq[batches])
plt.plot(epochs, acc_mapped[batches])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['xyz only', "with squares", "with mapped"], loc='upper left')
plt.show()
