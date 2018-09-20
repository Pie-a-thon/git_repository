from matplotlib import pyplot as plt
from trackml.dataset import load_event
import numpy as np
import pandas as pd
from skimage.transform import hough_circle, hough_circle_peaks, hough_line, hough_line_peaks
import random
import math as m
import scipy as sp

def load_mu(path_to_folder, number):
    list = np.linspace(1,5000, num=5000, dtype=int)

    samples = random.sample(set(list), number)

    hits = pd.DataFrame(columns = ["x", "y", "z", "labels", "other"])

    for i in samples:
        df=pd.read_csv(path_to_folder+"/clusters_"+str(i)+".csv", sep=',', names=["x", "y", "z", "labels", "other", "track"])
        df = df.fillna(0.0)
        sig = df.labels != 0.0
        df.labels[sig] = i
        hits = hits.append(df, ignore_index=True)

    return hits

def bin_plot(hits, num_bins=200, binary=False, mapped=False):
    if mapped:
        hist, x_edges, y_edges = np.histogram2d(getattr(hits, "v"), getattr(hits, "u"), bins=[num_bins, num_bins], range=[[-0.03,0.03],[-0.03,0.03]])
    else:
        hist, x_edges, y_edges = np.histogram2d(getattr(hits, "y"), getattr(hits, "x"), bins=[num_bins, num_bins], range=[[-1100,1100],[-1100,1100]])

    if binary:
        mask = hist != 0
        hist[mask] = 1
    return hist

def score(tracks, hits):
    score_list = []
    for hit_indices in tracks.candidate_hits:
        score = 0.0
        if len(hit_indices) == 0:
            score = 0.0
        else:
            true_hits = hits.loc[hits["labels"] != 0.0]
            particle_labels = np.unique(true_hits.labels.data)
            for label in particle_labels:
                particle_hits = true_hits.loc[true_hits["labels"] == label]
                match = len(set(hit_indices).intersection(particle_hits.index.values))
                if match/len(particle_hits.index.values) > 0.5 and match/len(hit_indices) > 0.5:
                    score = match/len(particle_hits)
    score_list.append(score)

    avg_score = sum(score_list)/len(score_list)
    return score_list, avg_score

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

    h_peak, theta_peak, d_peak = hough_line_peaks(h, theta, d, threshold=thresh)
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


data = load_mu("..\cernbox\inputs_ATLAS_step3_26082018/mu1GeV", 100)
hits, cells, particles, truth  = load_event("../train_sample/train_100_events/event000001000")

conformal_map(data)
conformal_map(hits)

x_0 = [8, 0.001, 0.0002]
def transform(X):
    thresh=int(X[0])
    max_from_center=X[1]
    max_dist_from_line=X[2]

    hist = bin_plot(data, num_bins=1000, binary=True, mapped=True)

    lines = get_lines(hist, num_bins = 1000, num_angles=800, thresh=thresh, max_dist=max_from_center)

    get_line_tracks(data, lines, max_dist=max_dist_from_line)

    score_list, avg_score = line_score(data, lines)
    res = sum((i-1.0)**2 for i in score_list) + abs(len(score_list) - 100)
    print(str(res) + " : "+str(score_list)+" : "+str(avg_score))

    return(res)

optimize = sp.optimize.minimize(transform, x0=[8, 0.001, 0.0002], method="TNC", bounds=[(0.0, None), (0.0, None), (0.0, None)])
print(optimize.x)


'''fig, ax = plt.subplots()
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
