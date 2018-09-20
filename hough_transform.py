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

def get_circles(hist, min_radius, max_radius, min_dist=100, num_radii=40, max_difference = 100.0, num_bins=200.0, num_peaks = np.inf, thresh=None):
    scaling = 2200/num_bins
    min_dist = int(min_dist)
    min_radius = min_radius/scaling
    max_radius = max_radius/scaling
    min_dist = int(min_dist/scaling)

    radii = np.linspace(min_radius,max_radius, num=num_radii)
    circles = hough_circle(hist, radii, full_output=True)
    accums, cx, cy, radius = hough_circle_peaks(circles, radii, normalize=True, min_xdistance=min_dist, min_ydistance=min_dist, total_num_peaks=num_peaks, threshold = thresh, num_peaks=num_peaks)
    scaled_cx = np.vectorize(lambda x: ((x-max_radius)-num_bins/2)*scaling)(cx)
    scaled_cy = np.vectorize(lambda y: ((y-max_radius)-num_bins/2)*scaling)(cy)
    scaled_radii = np.vectorize(lambda r: r*scaling)(radius)

    circles = pd.DataFrame(columns = ["cx", "cy", "radius"])
    for i in range(0, len(scaled_radii)):
        difference = abs(m.sqrt(scaled_cx[i]**2 + scaled_cy[i]**2) - scaled_radii[i])
        if difference < max_difference:
            circle = pd.DataFrame(np.column_stack([scaled_cx[i], scaled_cy[i], scaled_radii[i]]), columns = ["cx", "cy", "radius"])
            circles = circles.append(circle, ignore_index=True)

    return circles

def get_tracks(circs, hits_, num_tracks, tolerance_mult = 10.0, tolerance_add = 3.0, max_common = 1.0):
    tracks = pd.DataFrame(columns=["hit_indices", "cx", "cy", "radius", "num_hits", "score"])
    for i in range(0, len(getattr(circs, "index"))):
        score = 0.0
        theta = circs.cy[i]/circs.cx[i]
        above_hit_index_list = []
        below_hit_index_list = []
        radius = circs.radius[i]
        for j in range(0, len(hits_.index.values)):
            x = hits_.x.values[j]
            y = hits_.y.values[j]
            dx = abs(hits_.x.values[j] - circs.cx.values[i])
            dy = abs(hits_.y.values[j] - circs.cy.values[i])
            distance_sq = dx**2 + dy**2
            distance = m.sqrt(distance_sq)
            if radius - tolerance_mult < distance < radius+tolerance_mult:
                r = m.sqrt(x**2 + y**2)
                if radius - tolerance_mult*r/1000.0 - tolerance_add< distance < radius + tolerance_mult*r/1000.0 + tolerance_add:
                    if hits_.y.values[j] + hits_.x.values[j] * theta >= 0.0:
                        above_hit_index_list.append(hits_.index[j])
                    else:
                        below_hit_index_list.append(hits_.index[j])

        if len(above_hit_index_list) >= len(below_hit_index_list):
            hit_index_list = above_hit_index_list
        else:
            hit_index_list = below_hit_index_list

        num_hits = len(hit_index_list)

        track = pd.DataFrame({"hit_indices":[hit_index_list], "cx":[circs.cx[i]], "cy": [circs.cy[i]], "radius":[radius], "num_hits": [num_hits], "score": [score]})

        tracks = tracks.append(track, ignore_index=True)

    return tracks

def get_tracks_least_square(circs, hits_, tolerance_add, max_common=0.5, min_hits=10):
    candidate_tracks = pd.DataFrame(columns=["candidate_hits", "cx", "cy", "radius", "num_hits", "mean_square"])
    for i in hits_.index.values:
        hits_.track = [[] for _ in range(len(hits_))]

    for i in range(0, len(getattr(circs, "index"))):
        theta = circs.cy[i]/circs.cx[i]
        above_hit_index_list = []
        below_hit_index_list = []
        radius = circs.radius[i]
        above_mean_square_list = []
        below_mean_square_list = []
        for j in range(0, len(hits_.index.values)):
            x = hits_.x.values[j]
            y = hits_.y.values[j]
            dx = abs(hits_.x.values[j] - circs.cx.values[i])
            dy = abs(hits_.y.values[j] - circs.cy.values[i])
            distance_sq = dx**2 + dy**2
            distance = m.sqrt(distance_sq)
            if radius - tolerance_add< distance < radius + tolerance_add:
                if hits_.y.values[j] + hits_.x.values[j] * theta >= 0.0:
                    above_mean_square_list.append((distance - radius)**2)
                    above_hit_index_list.append(hits_.index[j])
                else:
                    below_mean_square_list.append((distance - radius)**2)
                    below_hit_index_list.append(hits_.index[j])

        if len(above_hit_index_list) >= len(below_hit_index_list):
            mean_square_list = above_mean_square_list
            hit_index_list = above_hit_index_list
        else:
            mean_square_list = below_mean_square_list
            hit_index_list = below_hit_index_list

        mean_square = sum(mean_square_list)/len(mean_square_list)

        num_hits = len(hit_index_list)

        track = pd.DataFrame({"candidate_hits":[hit_index_list], "cx":[circs.cx[i]], "cy": [circs.cy[i]], "radius":[radius], "num_hits": [num_hits], "mean_square": [mean_square]})

        candidate_tracks = candidate_tracks.append(track, ignore_index=True)


    candidates = list(candidate_tracks.candidate_hits.values)
    mean_squares = candidate_tracks.mean_square.values
    for i in range(0, len(candidates)):
        for j in range(0, len(candidates)):
            candidates = list(candidate_tracks.candidate_hits.values)
            if i !=j:
                common = list(set(candidates[i]).intersection(candidates[j]))
                if len(candidates[i]) != 0 and len(candidates[j]) != 0:
                    if len(common)/len(candidates[i]) > max_common and len(common)/len(candidates[j]) > max_common:
                        if mean_squares[i] > mean_squares[j]:
                            a, b = i, j
                        elif mean_squares[j] >= mean_squares[i]:
                            a, b = j, i

                        ic = np.isin(candidates[a], common)
                        indices = list(np.where(ic)[0])
                        candidate_tracks.candidate_hits[a] = np.delete(candidates[a], indices)

    tracks = pd.DataFrame(columns=["candidate_hits", "cx", "cy", "radius", "num_hits", "mean_square"])
    candidates = list(candidate_tracks.candidate_hits.values)
    for i in range(0, len(candidates)):
        if len(candidates[i]) >= min_hits:
            tracks = tracks.append(candidate_tracks.iloc[i], ignore_index=True)

    return tracks

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

def main():
    data = load_mu(".\cernbox\inputs_ATLAS_step3_26082018/mu1GeV", 30)
    plt.figure(1)
    plt.scatter(data.x, data.y, c = data.labels)
    plt.xlim(-1100,1100)
    plt.ylim(-1100,1100)

    tracks = pd.DataFrame(columns=["hit_indices", "cx", "cy", "radius", "num_hits", "score"])

    i=0
    reduced_data = data
    for i in range(0, 30):
        hist = bin_plot(reduced_data, num_bins=550)
        circles = get_circles(hist, 400, 1000, max_difference =30, min_dist=50, num_bins=550, num_peaks=1)
        track = get_tracks(circles, reduced_data, 1, tolerance_mult= 10.0, tolerance_add=5.0)
        if len(track.hit_indices.values) != 0.0:
            reduced_data = reduced_data.drop(track.hit_indices.values[0])
            print(reduced_data)
            tracks = tracks.append(track, ignore_index=True)
            i = i +1
        else:
            break

    for i in range(0, len(tracks.index.values)):
        print(tracks.hit_indices.values[i])
        #for j in tracks.hit_indices.values[i]:
        data.track[tracks.hit_indices.values[i]] = tracks.index.values[i]

    score_list, av_score = score(tracks, data)

    print(data)

    print(score_list)
    print(av_score)

    ####################

    #for making final plot
    plt.figure(2)
    plt.scatter(data.x[np.hstack(tracks.hit_indices.values)], data.y[np.hstack(tracks.hit_indices.values)], c = data.track[np.hstack(tracks.hit_indices.values)])
    plt.xlim(-1100,1100)
    plt.ylim(-1100,1100)

    plt.show()

#X_0 = [1600, 4000, 15, 80, 10, 0.0, 3]
def circle_test(x, plot = False):
    min_rad = int(x[0])
    max_rad = int(x[1])
    origin_dist = int(x[2])
    min_dist = int(x[3])
    tolerance = int(x[4])
    max_common = x[5]
    min_hits = int(x[6])

    circles = get_circles(hist, min_rad, max_rad, max_difference = origin_dist, min_dist= min_dist, num_bins=550, num_peaks=200)
    tracks = get_tracks_least_square(circles, data, tolerance, max_common=max_common, min_hits=min_hits)

    if plot:
        for i in range(0, len(tracks.index.values)):
            data.track[tracks.candidate_hits.values[i]] = tracks.index.values[i]

        fig, ax = plt.subplots()
        for i in tracks.index.values:
            circle = plt.Circle((tracks.cx.values[i], tracks.cy.values[i]), tracks.radius.values[i], color='r', fill=False, linewidth = 1)
            ax.add_artist(circle)


        plt.figure(1)
        plt.scatter(data.x, data.y, c = data.labels)
        plt.xlim(-1100,1100)
        plt.ylim(-1100,1100)

        plt.figure(2)
        plt.scatter(data.x[np.hstack(tracks.candidate_hits.values)], data.y[np.hstack(tracks.candidate_hits.values)], c = data.track[np.hstack(tracks.candidate_hits.values)])
        plt.xlim(-1100,1100)
        plt.ylim(-1100,1100)
        plt.show()

    score_list, av_score = score(tracks, data)
    res = sum((i-1.0)**2 for i in score_list) + abs(len(score_list) - 10)
    print(str(res) + " : "+str(score_list)+" : "+str(av_score))
    return res

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

    print(lines)
    if max_dist>=0.0:
        within_dist = abs(lines.origin_distance.values) <= max_dist
        lines = lines[within_dist]
        lines.reset_index(drop=True, inplace=True)

    print(lines)
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


data = load_mu(".\cernbox\inputs_ATLAS_step3_26082018/mu1GeV", 100)
hits, cells, particles, truth  = load_event("./train_sample/train_100_events/event000001000")


track_hits = truth.particle_id == 900720956266250240

conformal_map(data)
conformal_map(hits)

hist = bin_plot(data, num_bins=500, binary=True, mapped=True)

lines = get_lines(hist, num_bins = 500, num_angles=800, max_dist=0.001)

get_line_tracks(data, lines, max_dist=0.0001)

print(data)

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

plt.show()











'''fig, ax = plt.subplots()
for angle, dist in zip(lines.angle.values, lines.origin_distance.values):
    y0 = (dist - (-0.03) *np.cos(angle))/ np.sin(angle)
    y1 = (dist - (0.03) *np.cos(angle))/ np.sin(angle)
    ax.plot((-0.03, 0.03), (y0, y1), '-r')
    ax.set_xlim((-0.03, 0.03))
    ax.set_ylim((-0.03, 0.03))
    ax.set_title('Detected lines')

    plt.scatter(data.u, data.v, c=data.labels)
    '''




'''
fig, ax = plt.subplots()
for i in circles.index.values:
    circle = plt.Circle((circles.cx.values[i], circles.cy.values[i]), circles.radius.values[i], color='r', fill=False, linewidth = 1)
    ax.add_artist(circle)'''
