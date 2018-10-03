from trackml.dataset import load_event
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import math as m
import tensorflow as tf
from tensorflow import keras
import pickle

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_mu(path_to_folder, number):
    list = np.linspace(1,5000, num=5000, dtype=int)

    samples = random.sample(set(list), number)

    hits = pd.DataFrame(columns = ["x", "y", "z", "labels", "other"])

    for i in samples:
        df=pd.read_csv(path_to_folder+"/clusters_"+str(i)+".csv", sep=',', names=["x", "y", "z", "labels", "other"])
        df = df.fillna(0.0)
        sig = df.labels != 0.0
        df.labels[sig] = i
        hits = hits.append(df, ignore_index=True)

    return hits


def conformal_map(hits, x0=0, y0=0):
    x = hits.x.values
    y = hits.y.values
    u_list = []
    v_list = []
    phi_list = []
    rho_list = []
    for i in range (0, len(hits.index.values)):
        r_sq = ((x[i]-x0)**2 + (y[i]-y0)**2)
        u = (x[i]-x0)/r_sq
        v = (y[i]-y0)/r_sq
        phi = np.arctan2(v, u)
        rho = m.sqrt(u**2 + v**2)
        d = u*np.cos(phi) + v*np.sin(phi)
        u_list.append(u)
        v_list.append(v)
        phi_list.append(phi)
        rho_list.append(rho)

    hits["u"] = pd.Series(u_list, index = hits.index)
    hits["v"] = pd.Series(v_list, index = hits.index)
    hits["phi"] = pd.Series(phi_list, index = hits.index)
    hits["rho"] = pd.Series(rho_list, index = hits.index)

def muons():

    data = load_mu("..\cernbox\inputs_ATLAS_step3_26082018/mu1GeV", 3)
    features = (data[['x','y','z']]/1000.0)
    particle_ids = data.labels.unique()
    particle_ids = particle_ids[np.where(particle_ids!=0)[0]]

    Train = []
    pair = []
    for particle_id in particle_ids:
        hit_ids = data.index.values[data.labels == particle_id]
        for i in hit_ids:
            for j in hit_ids:
                if i != j:
                    pair.append([i,j])

    pair = np.array(pair)
    Train1 = np.hstack((features.values[pair[:,0]], features.values[pair[:,1]], np.ones((len(pair),1))))

    if len(Train) == 0:
            Train = Train1
    else:
        Train = np.vstack((Train,Train1))

    n = len(data)
    size = len(Train1)*3
    p_id = data.labels.values
    i =np.random.randint(n, size=size)
    j =np.random.randint(n, size=size)
    pair = np.hstack((i.reshape(size,1),j.reshape(size,1)))
    pair = pair[((p_id[i]==0) | (p_id[i]!=p_id[j]))]

    Train0 = np.hstack((features.values[pair[:,0]], features.values[pair[:,1]], np.zeros((len(pair),1))))

    Train = np.vstack((Train,Train0))

    np.random.shuffle(Train)
    print(Train)
    return Train

def get_t_tbar(path, add_squares=False, mapped=False):
    hits = pd.read_csv(path, sep=',', names=["x", "y", "z", "particle_id", "particle_id_1", "particle_id_2"])
    hits = hits.fillna(0.0)
    squares = (hits[['x','y','z']]/1000)**2
    squares.columns=["x^2", "y^2", "z^2"]
    if mapped:
        conformal_map(hits)
        map_vals = hits[["u", "v", "phi"]]
    truth = hits
    if add_squares:
        if mapped:
            features = pd.concat([hits[['x','y','z']]/1000, squares, map_vals], axis=1)
        else:
            features = pd.concat([hits[['x','y','z']]/1000, squares], axis=1)

    else:
        if mapped:
            features = pd.concat([hits[['x','y','z']]/1000, map_vals], axis=1)
        else:
            features = (hits[['x','y','z']]/1000)


    print(features)

    particle_ids = truth.particle_id.unique()
    particle_ids = particle_ids[np.where(particle_ids!=0)[0]]
    Train=[]
    pair = []
    for particle_id in particle_ids:
        hit_ids = truth[truth.particle_id == particle_id].index.values
        for i in hit_ids:
            for j in hit_ids:
                if i != j:
                    pair.append([i,j])

    pair = np.array(pair)

    Train1 = np.hstack((features.values[pair[:,0]], features.values[pair[:,1]], np.ones((len(pair),1))))

    if len(Train) == 0:
        Train = Train1
    else:
        Train = np.vstack((Train,Train1))

    n = len(hits)
    size = len(Train1)*3
    p_id = truth.particle_id.values
    i =np.random.randint(n, size=size)
    j =np.random.randint(n, size=size)
    pair = np.hstack((i.reshape(size,1),j.reshape(size,1)))
    pair = pair[((p_id[i]==0) | (p_id[i]!=p_id[j]))]

    Train0 = np.hstack((features.values[pair[:,0]], features.values[pair[:,1]], np.zeros((len(pair),1))))

    Train = np.vstack((Train,Train0))
    del Train0, Train1

    np.random.shuffle(Train)
    print(Train.shape)

    return Train

def get_train_data(path_to_event, add_cells = True, add_squares= False):
    hits, cells, particles, truth = load_event(path_to_event) #"../train_sample/train_100_events/event000001000"
    hit_cells = cells.groupby(['hit_id']).value.count().values
    hit_value = cells.groupby(['hit_id']).value.sum().values
    squares = (hits[['x','y','z']]/1000)**2
    squares.columns=["x^2", "y^2", "z^2"]
    if add_cells:
        if add_squares:
            features = np.hstack([pd.concat([hits[['x','y','z']]/1000, squares], axis=1), hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)])
            print(features)
        else:
            features = np.hstack([hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)])

    else:
        if add_squares:
            features = pd.concat([hits[['x','y','z']]/1000, squares], axis=1)
            print(features)

        else:
            features = (hits[['x','y','z']]/1000)

    particle_ids = truth.particle_id.unique()
    particle_ids = particle_ids[np.where(particle_ids!=0)[0]]
    Train=[]
    pair = []
    for particle_id in particle_ids:
        hit_ids = truth[truth.particle_id == particle_id].hit_id.values-1
        for i in hit_ids:
            for j in hit_ids:
                if i != j:
                    pair.append([i,j])

    pair = np.array(pair)
    if add_cells:
            Train1 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.ones((len(pair),1))))
    else:
        Train1 = np.hstack((features.values[pair[:,0]], features.values[pair[:,1]], np.ones((len(pair),1))))

    if len(Train) == 0:
        Train = Train1
    else:
        Train = np.vstack((Train,Train1))

    n = len(hits)
    size = len(Train1)*3
    p_id = truth.particle_id.values
    i =np.random.randint(n, size=size)
    j =np.random.randint(n, size=size)
    pair = np.hstack((i.reshape(size,1),j.reshape(size,1)))
    pair = pair[((p_id[i]==0) | (p_id[i]!=p_id[j]))]

    if add_cells:
        Train0 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.zeros((len(pair),1))))
    else:
        Train0 = np.hstack((features.values[pair[:,0]], features.values[pair[:,1]], np.zeros((len(pair),1))))

    Train = np.vstack((Train,Train0))
    del Train0, Train1

    np.random.shuffle(Train)
    print(Train.shape)

    return Train

def train_new_model(model_name, Train, with_cells=False, with_squares=False, with_mapped=False):
    h1 = LossHistory()
    h2 = LossHistory()
    h3 = LossHistory()

    num_inputs = 6
    if with_cells:
        num_inputs = num_inputs + 4
    if with_squares:
        num_inputs = num_inputs + 6
    if with_mapped:
        num_inputs = num_inputs + 6

    model = keras.Sequential([
        keras.layers.Dense(num_inputs),
        keras.layers.Dense(800, activation="selu"),
        keras.layers.Dense(400, activation="selu"),
        keras.layers.Dense(400, activation="selu"),
        keras.layers.Dense(400, activation="selu"),
        keras.layers.Dense(200, activation="selu"),
        keras.layers.Dense(1, activation = "sigmoid")
        ])

    lr=-5
    model.compile(loss=['binary_crossentropy'], optimizer=keras.optimizers.Adam(lr=10**(lr)), metrics=['accuracy'], do_validation=True)
    History_1 = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=1, verbose=1, validation_split=0.05, shuffle=True,callbacks=[h1])

    lr=-4
    model.compile(loss=['binary_crossentropy'], optimizer=keras.optimizers.Adam(lr=10**(lr)), metrics=['accuracy'])
    History_2 = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=20, verbose=1, validation_split=0.05, shuffle=True,callbacks=[h2])

    lr=-5
    model.compile(loss=['binary_crossentropy'], optimizer=keras.optimizers.Adam(lr=10**(lr)), metrics=['accuracy'])
    History_3 = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=3, verbose=1, validation_split=0.05, shuffle=True,callbacks=[h3])

    model.save(model_name)

    return h1, h2, h3

def load_and_train(path, data, save_path=None):
    h1 = LossHistory()
    h2 = LossHistory()
    h3 = LossHistory()
    model = keras.models.load_model(path)
    model.summary()
    Train = data

    lr=-5
    History_1 = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=1, verbose=1, validation_split=0.05, shuffle=True,callbacks=[h1])

    lr=-4
    History_2 = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=20, verbose=1, validation_split=0.05, shuffle=True,callbacks=[h2])

    lr=-5
    History_3 = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=3, verbose=1, validation_split=0.05, shuffle=True,callbacks=[h3])

    if save_path!= None:
        model.save(save_path)

    return h1, h2, h3

def test_model(path_to_model, data):
    Train = data
    new_model = keras.models.load_model(path_to_model) #'../my_model.h5'
    new_model.summary()
    test = new_model.evaluate(x=Train[:,:-1], y=Train[:,-1],verbose = 1)
    print(new_model.metrics_names)
    print(test)

def get_predictions(path_to_model, data):
    Train = data
    new_model = keras.models.load_model(path_to_model) #'../my_model.h5'
    new_model.summary()
    predictions = new_model.predict(x=Train[:,:-1], y=Train[:,-1],verbose = 1)
    print(predictions)

#################################################################################################################
#################################################################################################################
def test_on_event(event, existing_model):
    hits = get_t_tbar("..\cernbox\inputs_ATLAS_step3_26082018/ttbar\clusters_277000"+str(event)+".csv",mapped=False, add_squares=False)
    if existing_model:
        h1, h2, h3 = load_and_train("my_model_ttbar.h5", hits, "my_model_ttbar.h5")
    else:
        h1, h2, h3 = train_new_model("my_model_ttbar.h5", hits)
    save_obj(np.hstack([h1.acc, h2.acc, h3.acc]), "accuracy_ttbar_"+str(event))
    save_obj(np.hstack([h1.losses, h2.losses, h3.losses]), "losses_ttbar_"+str(event))

    hits = get_t_tbar("..\cernbox\inputs_ATLAS_step3_26082018/ttbar\clusters_277000"+str(event)+".csv",mapped=False, add_squares=True)
    if existing_model:
        h1, h2, h3 = load_and_train("my_model_ttbar_squares.h5", hits, "my_model_ttbar_squares.h5")
    else:
        h1, h2, h3 = train_new_model("my_model_ttbar_squares.h5", hits, with_squares=True)
    save_obj(np.hstack([h1.acc, h2.acc, h3.acc]), "accuracy_ttbar_"+str(event)+"_squares")
    save_obj(np.hstack([h1.losses, h2.losses, h3.losses]), "losses_ttbar_"+str(event)+"_squares")

    hits = get_t_tbar("..\cernbox\inputs_ATLAS_step3_26082018/ttbar\clusters_277000"+str(event)+".csv",mapped=True, add_squares=False)
    if existing_model:
        h1, h2, h3 = load_and_train("my_model_ttbar_mapped.h5", hits, "my_model_ttbar_mapped.h5")
    else:
        h1, h2, h3 = train_new_model("my_model_ttbar_mapped.h5", hits, with_mapped=True)
    save_obj(np.hstack([h1.acc, h2.acc, h3.acc]), "accuracy_ttbar_"+str(event)+"_mapped")
    save_obj(np.hstack([h1.losses, h2.losses, h3.losses]), "losses_ttbar_"+str(event)+"_mapped")

test_on_event(1, existing_model=False)
test_on_event(2, existing_model=True)
test_on_event(3, existing_model=True)

'''h1, h2, h3 = load_and_train("my_model_ttbar.h5", hits, "my_model_ttbar.h5")
save_obj(np.hstack([h1.acc, h2.acc, h3.acc]), "accuracy_ttbar_3")
save_obj(np.hstack([h1.losses, h2.losses, h3.losses]), "losses_ttbar_3")

h1, h2, h3 = load_and_train("my_model_ttbar_squares.h5", hits, "my_model_ttbar_squares.h5")
save_obj(np.hstack([h1.acc, h2.acc, h3.acc]), "accuracy_ttbar_3_squares")
save_obj(np.hstack([h1.losses, h2.losses, h3.losses]), "losses_ttbar_3_squares")'''

'''data = get_train_data("../train_sample/train_100_events/event000001001", add_cells =True, add_squares=False)
h1, h2, h3 = load_and_train("my_model_with_cells.h5", data, save_path="my_model_with_cells_1.h5")
save_obj(np.hstack([h1.acc, h2.acc, h3.acc]), "accuracy_cells_1")
save_obj(np.hstack([h1.losses, h2.losses, h3.losses]), "losses_cells_1")'''

'''data = get_train_data("../train_sample/train_100_events/event000001001", add_cells =True)
print("with cells")
test_model("my_model_with_cells.h5", data)


data = get_train_data("../train_sample/train_100_events/event000001001", add_cells =False)
print("without cells")
test_model("my_model.h5", data)
'''
