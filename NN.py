from trackml.dataset import load_event
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import math as m
import tensorflow as tf
from tensorflow import keras

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

def muons():

    data = load_mu("..\cernbox\inputs_ATLAS_step3_26082018/mu1GeV", 100)
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
'''
Train0 = np.hstack((features.values[pair[:,0]], features.values[pair[:,1]], np.zeros((len(pair),1))))

Train = np.vstack((Train,Train0))

np.random.shuffle(Train)
print(Train)'''
#################################################################################################################
#################################################################################################################

hits, cells, particles, truth = load_event("../train_sample/train_100_events/event000001000")
hit_cells = cells.groupby(['hit_id']).value.count().values
hit_value = cells.groupby(['hit_id']).value.sum().values
features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))
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
Train1 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.ones((len(pair),1))))

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

Train0 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.zeros((len(pair),1))))

Train = np.vstack((Train,Train0))
del Train0, Train1

np.random.shuffle(Train)
print(Train.shape)




'''input_list = []
truth_list = []

for i in range(0, len(data.x.values)):
    for j in range(0, len(data.x.values)):
        input = [data.x.values[i], data.y.values[i], data.z.values[i], data.x.values[j], data.y.values[j], data.z.values[j]]
        if data.labels.values[i] == data.labels.values[j]:
            truth = 1.0
        else:
            truth = 0.0

        input_list.append(input)
        truth_list.append(truth)

input_array = np.array(input_list)
truth_array = np.array(truth_list)
'''

model = keras.Sequential([
    keras.layers.Dense(6),
    keras.layers.Dense(800, activation="selu"),
    keras.layers.Dense(400, activation="selu"),
    keras.layers.Dense(400, activation="selu"),
    keras.layers.Dense(400, activation="selu"),
    keras.layers.Dense(200, activation="selu"),
    keras.layers.Dense(1, activation = "sigmoid") #activation="sigmoid"
])

lr=-5
model.compile(loss=['binary_crossentropy'], optimizer=keras.optimizers.Adam(lr=10**(lr)), metrics=['accuracy'])
History = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=1, verbose=2, validation_split=0.05, shuffle=True)

lr=-4
model.compile(loss=['binary_crossentropy'], optimizer=keras.optimizers.Adam(lr=10**(lr)), metrics=['accuracy'])
History = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=20, verbose=2, validation_split=0.05, shuffle=True)

lr=-5
model.compile(loss=['binary_crossentropy'], optimizer=keras.optimizers.Adam(lr=10**(lr)), metrics=['accuracy'])
History = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=3, verbose=2, validation_split=0.05, shuffle=True)


model.save('my_model_with_cells.h5')


'''new_model = keras.models.load_model('../my_model.h5')
new_model.summary()
test = new_model.evaluate(x=Train[:,:-1], y=Train[:,-1],verbose = 1)
print(new_model.metrics_names)
print(test)'''

'''predictions = new_model.predict(input_array)
print(predictions)'''
