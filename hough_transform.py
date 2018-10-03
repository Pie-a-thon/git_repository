from matplotlib import pyplot as plt
from trackml.dataset import load_event
import numpy as np
import pandas as pd
from skimage.transform import hough_circle, hough_circle_peaks, hough_line, hough_line_peaks
import random
import math as m
import scipy as sp
import pickle

def load_mu(path_to_folder, number, full_out=False):
    list = np.linspace(1,5000, num=5000, dtype=int)

    samples = random.sample(set(list), number)

    hits = pd.DataFrame(columns = ["x", "y", "z", "labels", "other"])
    truth = pd.DataFrame(columns=["barcode", "pt", "eta", "phi", "e", "labels"])

    for i in samples:
        df=pd.read_csv(path_to_folder+"/clusters_"+str(i)+".csv", sep=',', names=["x", "y", "z", "labels", "other", "track"])
        df_truth=pd.read_csv(path_to_folder+"/truth_"+str(i)+".csv", sep=',', names=["barcode", "pt", "eta", "phi", "e", "labels"])
        row = (df_truth.iloc[0].values)
        df_truth = pd.DataFrame({"barcode":row[0], "pt":row[1], "eta":row[2], "phi":row[3], "e":row[4], "labels":row[5]}, index=[1])
        df = df.fillna(0.0)
        sig = df.labels != 0.0
        df.labels[sig] = i
        hits = hits.append(df, ignore_index=True)
        truth = truth.append(df_truth)

    truth.reset_index(drop=True, inplace=True)
    if full_out:
        return hits, truth

    else:
        return hits

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def bin_plot(hits, num_bins=200, binary=False, mapped=False):
    if mapped:
        hist, x_edges, y_edges = np.histogram2d(getattr(hits, "v"), getattr(hits, "u"), bins=[num_bins, num_bins], range=[[-0.03,0.03],[-0.03,0.03]])
    else:
        hist, x_edges, y_edges = np.histogram2d(getattr(hits, "y"), getattr(hits, "x"), bins=[num_bins, num_bins], range=[[-1100,1100],[-1100,1100]])

    if binary:
        mask = hist != 0
        hist[mask] = 1
    return hist

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

def get_lines(hist, num_bins, num_angles=1000, thresh=None, max_dist=-1):
    scaling = 0.06/num_bins
    theta = np.linspace(-np.pi/2, np.pi/2, num=num_angles)
    h, theta, d = hough_line(hist, theta=theta)

    h_peak, theta_peak, d_peak = hough_line_peaks(h, theta, d, threshold=thresh, min_distance=20)
    gradients = np.vectorize(lambda a: -np.tan((np.pi/2 - a)))(theta_peak)

    scaled_intercept_list = []
    for angle, dist in zip(theta_peak, d_peak):
        intercept = (dist-(num_bins/2)*np.cos(angle)) / np.sin(angle)
        scaled_intercept = scaling*intercept -0.03
        scaled_intercept_list.append(scaled_intercept)

    scaled_dist = []
    for angle, c, m in zip(theta_peak, scaled_intercept_list, gradients):
        d = c/(np.sin(angle) - m*np.cos(angle))
        scaled_dist.append(d)

    lines = pd.DataFrame(np.column_stack([theta_peak, d_peak, gradients, scaled_intercept_list, scaled_dist]), columns = ["angle", "histogram_distance", "gradient", "intercept", "origin_distance"])

    if max_dist>=0.0:
        within_dist = abs(lines.origin_distance.values) <= max_dist
        lines = lines[within_dist]
        lines.reset_index(drop=True, inplace=True)

    return lines

def get_line_tracks(hits, lines, max_dist=None):
    if max_dist != None:
        max_dist_sq = max_dist**2
    else:
         max_dist_sq = np.inf

    for i in range(0, len(hits.index.values)):
        u = hits.u.values[i]
        v = hits.v.values[i]
        phi = hits.phi.values[i]

        dist_square_list = []
        for j in range(0, len(lines.index.values)):
            m = lines.gradient.values[j]
            c = lines.intercept.values[j]

            dist_sq = (v - (c+m*u))**2 / (1.0+m**2)
            dist_square_list.append(dist_sq)

        minimum = min(dist_square_list)

        if minimum <= max_dist_sq:
            track = np.where(dist_square_list == minimum)[0]

        else:
            track = -1.0

        #val, idx = min((val, idx) for (idx, val) in enumerate(dist_square_list))
        hits.track[i] = track

def line_score(hits, lines):
    score_list = []
    for i in range(0, len(lines)):
        score = 0.0
        track_indices = hits.loc[(hits["track"] == i)].index.values
        if len(track_indices) == 0:
            score = 0.0
        else:
            true_hits = hits.loc[hits["labels"] >= 0.0]
            particle_labels = np.unique(true_hits.labels.data)
            for label in particle_labels:
                particle_hits = true_hits.loc[true_hits["labels"] == label]
                match = len(set(track_indices).intersection(particle_hits.index.values))
                if match/len(particle_hits.index.values) > 0.5 and match/len(track_indices) > 0.5:
                    score = match/len(particle_hits)

        score_list.append(score)

    avg_score = sum(score_list)/len(score_list)
    return score_list, avg_score

def line_test(hist, num_bins, num_angles, num_hits):
    lines = get_lines(hist, num_bins=num_bins, num_angles=num_angles, max_dist=0.001)
    res = (num_hits-len(lines.index.values))**2
    return res

def lines_optim(min=100, max=400, num=10):
    # columns=num_bins, rows=num_angles
    num_hits = np.linspace(min, max, num =num)
    for i in num_hits:
        data = load_mu("..\cernbox\inputs_ATLAS_step3_26082018/mu1GeV", int(i))
        conformal_map(data)
        columns = np.linspace(500, 2000, num=20)
        rows = np.linspace(500, 2000, num = 20)

        heatmap = pd.DataFrame(0.0, columns=columns, index=rows)
        for j in columns:
            for k in rows:
                hist = bin_plot(data, num_bins=j, binary=True, mapped=True)
                cell = line_test(hist, j, k, int(i))
                heatmap[j][k] = cell
                print(heatmap)

                heatmap.to_csv("bins_angles_"+str(int(i))+".csv")

def avg_number():
    heatmap = pd.read_csv("bins_angles_100.csv", sep=',', index_col =0)


    for i in np.linspace(100, 400, num=10):
        df = pd.read_csv("bins_angles_"+str(int(i))+".csv", sep=',', index_col =0)
        heatmap = heatmap + df

    print(heatmap)

    plt.imshow(heatmap, interpolation="gaussian")
    plt.show()

    print(heatmap.min(1), heatmap.min(0))

def eta_test():
    eta_list = []
    scores = []
    for i in range(0, 200):
        data, truth = load_mu("..\cernbox\inputs_ATLAS_step3_26082018/mu1GeV", 5, full_out=True)
        conformal_map(data)
        hist = bin_plot(data, num_bins=500, binary=True, mapped=True)
        lines = get_lines(hist, num_bins=500, num_angles=900)
        tracks = get_line_tracks(data, lines)
        score_list, avg_score = line_score(data, lines)
        print(score_list)
        print(avg_score)

        avg_eta = sum(truth.eta.values)/len(truth.eta.values)
        scores.append(avg_score)
        eta_list.append(avg_eta)

    print(sum(scores)/len(scores))
    plt.scatter(eta_list, scores)
    plt.show()


hits, cells, particles, truth  = load_event("../train_sample/train_100_events/event000001000")


'''data = load_mu("..\cernbox\inputs_ATLAS_step3_26082018/mu1GeV", 1)
conformal_map(data)
columns = np.linspace(500, 2000, num = 20)
rows = np.linspace(500, 2000, num = 20)

heatmap = pd.DataFrame(0.0, columns=columns, index=rows)
for j in columns:
    for k in rows:
        hist = bin_plot(data, num_bins=j, binary=True, mapped=True)
        cell = line_test(hist, j, k, 1)
        heatmap[j][k] = cell
        print(heatmap)

        heatmap.to_csv("bins_angles_"+str(int(i))+".csv")

plt.imshow(heatmap)
plt.show()'''

def muon_efficiency():
    scores = []
    avg_scores = []
    run_list = []
    for i in range(1, 100):
        df=pd.read_csv("..\cernbox\inputs_ATLAS_step3_26082018/mu100GeV/clusters_"+str(i)+".csv", sep=',', names=["x", "y", "z", "labels", "other", "track"])
        df_truth=pd.read_csv("..\cernbox\inputs_ATLAS_step3_26082018/mu100GeV/truth_"+str(i)+".csv", sep=',', names=["barcode", "pt", "eta", "phi", "e", "labels"])
        row = (df_truth.iloc[0].values)
        df_truth = pd.DataFrame({"barcode":row[0], "pt":row[1], "eta":row[2], "phi":row[3], "e":row[4], "labels":row[5]}, index=[1])
        df = df.fillna(0.0)
        conformal_map(df)
        hist = bin_plot(df, num_bins=200, mapped=True)
        lines = get_lines(hist, num_bins=200, num_angles=100, max_dist=0.0007)
        get_line_tracks(df, lines)
        score_list, avg_score = line_score(df, lines)
        scores.append(avg_score)
        average = sum(scores)/len(scores)
        avg_scores.append(average)
        run_list.append(i)

    print(lines)
    print(avg_scores)

    save_obj(scores, "muon_scores")
    save_obj(avg_scores, "muon_avg_scores")
    save_obj(run_list, "muon_runs")

    plt.figure(3)
    fig, ax = plt.subplots()
    for m, c in zip(lines.gradient.values, lines.intercept.values):
            y0 = m*(-0.03) + c
            y1 = m*(0.03) + c
            ax.plot((-0.03, 0.03), (y0, y1), '-r')
    ax.set_xlim((-0.03, 0.03))
    ax.set_ylim((-0.03, 0.03))
    ax.set_title('Detected lines')

    plt.scatter(df.u, df.v, c=df.labels)

    plt.figure(1)
    plt.plot(run_list, scores)
    plt.title('single particle score')
    plt.ylabel('score')
    plt.xlabel('run')

    plt.figure(2)
    plt.plot(run_list, avg_scores)
    plt.title('single particle avg score')
    plt.ylabel('avg score')
    plt.xlabel('run')

    plt.show()

scores = []
avg_scores = []
resolution_list = []
eta_list = []
run_list = []
for i in range(1, 100):
    df=pd.read_csv("..\cernbox\inputs_ATLAS_step3_26082018/mu1GeV/clusters_"+str(i)+".csv", sep=',', names=["x", "y", "z", "labels", "other", "track"])
    df_truth=pd.read_csv("..\cernbox\inputs_ATLAS_step3_26082018/mu1GeV/truth_"+str(i)+".csv", sep=',', names=["barcode", "pt", "eta", "phi", "e", "labels"])
    row = (df_truth.iloc[0].values)
    df_truth = pd.DataFrame({"barcode":row[0], "pt":row[1], "eta":row[2], "phi":row[3], "e":row[4], "labels":row[5]}, index=[1])
    df = df.fillna(0.0)
    conformal_map(df)
    hist = bin_plot(df, num_bins=200, mapped=True)
    lines = get_lines(hist, num_bins=200, num_angles=100, max_dist=0.0007)
    get_line_tracks(df, lines)
    gradient = lines.gradient.values[0]
    intercept = lines.intercept.values[0]
    momentum = 0.3*2.0*(gradient**2 +1)/(4*intercept**2)
    eta = abs(df_truth.eta.values[0])
    resolution = momentum# - df_truth.pt.values[0])/momentum
    resolution_list.append(resolution)
    eta_list.append(eta)
    run_list.append(i)

plt.scatter(eta_list, resolution_list)
plt.title('resolution and eta')
plt.ylabel('resolution %')
plt.xlabel('Eta')
plt.show()


'''conformal_map(hits)

X = [8, 0.001, 0.0002]
thresh=int(X[0])
max_from_center=X[1]
max_dist_from_line=X[2]


lines = get_lines(hist, num_bins = 1053, num_angles=1431, thresh=thresh, max_dist=max_from_center)'''

'''data = load_mu("..\cernbox\inputs_ATLAS_step3_26082018/mu1GeV", 400)
conformal_map(data)
hist = bin_plot(data, num_bins=1600, binary=True, mapped=True)
print(line_test(hist, 1600, 3600, 400))'''


#lines_optim()


#tracks = get_line_tracks(data, lines)

'''def tracks(thresh):
    lines = get_lines(hist, num_bins = 1000, num_angles=800, thresh=int(thresh), max_dist=0.001)
    get_line_tracks(data, lines, max_dist=0.0001)

    score_list, avg_score = line_score(data, lines)
    res = sum((i-1.0)**2 for i in score_list) + abs(len(score_list) - 100)
    print(thresh)
    print(str(res) + " : "+str(score_list)+" : "+str(avg_score))
    return res

thresholds = np.linspace(0, 20, num=20)
scores = np.vectorize(tracks)(thresholds)

plt.plot(thresholds, scores)
plt.show()'''


#optimize = sp.optimize.minimize(transform, x0=[8, 0.001, 0.0002], method="Nelder-Mead", bounds=[(0.0, None), (0.0, None), (0.0, None)])
#print(optimize.x)


'''print(lines)

score_list, avg_score = line_score(data, lines)
print(score_list)
print(avg_score)

fig, ax = plt.subplots()
for m, c in zip(lines.gradient.values, lines.intercept.values):
        y0 = m*(-0.03) + c
        y1 = m*(0.03) + c
        ax.plot((-0.03, 0.03), (y0, y1), '-r')
ax.set_xlim((-0.03, 0.03))
ax.set_ylim((-0.03, 0.03))
ax.set_title('Detected lines')

plt.scatter(data.u, data.v, c=data.labels)

plt.figure(3)
plt.scatter(data.u, data.v, c=data.track)

fig, ax = plt.subplots()
ax.imshow(hist)
for angle, dist in zip(lines.angle.values, lines.histogram_distance.values):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - hist.shape[1] * np.cos(angle)) / np.sin(angle)
        ax.plot((0, hist.shape[1]), (y0, y1), '-r')
ax.set_xlim((0, hist.shape[1]))
ax.set_ylim((0, hist.shape[0]))
ax.set_title('Detected lines')

plt.show()'''
