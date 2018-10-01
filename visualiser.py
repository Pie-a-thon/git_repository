from matplotlib import pyplot as plt
import pickle
import numpy as np

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

losses = load_obj("losses_cells_1")
losses_no_cells = load_obj("losses_no_cells_1")
#losses_squares = load_obj("losses_squares")

acc = load_obj("accuracy_cells_1")
acc_no_cells = load_obj("accuracy_no_cells_1")
#acc_squares = load_obj("accuracy_squares")


print(len(losses)/24)
batches = np.linspace(511.0, len(losses)-1, num = 24, dtype=int)
print(batches)
epochs = np.linspace(0, 24, num=24)

plt.plot(epochs, losses[batches])
plt.plot(epochs, losses_no_cells[batches])
#plt.plot(epochs, losses_squares[batches])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['With cells', 'Without cells', "with squares"], loc='upper left')

plt.figure(2)
plt.plot(epochs, acc[batches])
plt.plot(epochs, acc_no_cells[batches])
#plt.plot(epochs, acc_squares[batches])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['With cells', 'Without cells', "with squares"], loc='upper left')
plt.show()
