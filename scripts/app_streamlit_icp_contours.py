#!/usr/bin/env python
"""
app_streamlit_icp_contours.py

Cette application Streamlit permet de :
  - Charger deux maillages 3D (source et cible) et leurs repères (.bnd).
  - Extraire les repères correspondant aux bords intérieurs des yeux (via indices fournis ou par défaut).
  - Calculer la représentation bipolaire (potentiels géodésiques) en utilisant ces repères.
  - Extraire des contours (level sets) sur le maillage via des distances cibles (entrées manuelles) et une tolérance.
  - Ordonner, rééchantillonner et interpoler les contours avec des B‑splines.
    * Pour le level set "sum" : une unique courbe est obtenue.
    * Pour le level set "diff" : les deux branches (gauche et droite) sont traitées séparément.
  - Aligner les contours par ICP, soit pour une paire sélectionnée, soit en mode batch.
  - Visualiser les résultats d'alignement et les métriques ICP.

Auteur : Malek DINARI
"""

import streamlit as st
import numpy as np
import trimesh
import torch
import heapq
import time
from tqdm import tqdm
from scipy.sparse import lil_matrix
import open3d as o3d
import matplotlib.pyplot as plt
import tempfile
import plotly.graph_objs as go
from io import StringIO
from scipy.interpolate import splprep, splev  # Pour B-spline

st.title("Alignement ICP sur les contours bipolaires")
st.markdown("""
Cette application vous permet d’aligner deux maillages 3D (source et cible) à l’aide de la représentation 
bipolaire des potentiels géodésiques et de l’algorithme ICP appliqué aux contours extraits (level sets).  
Les repères utilisés correspondent aux bords intérieurs des yeux.
""")

#######################################
# Définir l'appareil CUDA si disponible
#######################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write("Appareil utilisé :", device)

#######################################
# Fonctions pour la lecture des repères (.bnd) et extraction des repères
#######################################
def read_bnd_file(file_bytes):
    try:
        text = file_bytes.decode("utf-8")
    except Exception as e:
        st.error(f"Erreur lors du décodage du fichier .bnd: {e}")
        return None

    st.write("Extrait des premières lignes du fichier .bnd :")
    lines = text.splitlines()
    for i, line in enumerate(lines[:5]):
        st.write(f"Ligne {i}: {line}")

    try:
        s = StringIO(text)
        data = np.genfromtxt(s, delimiter=None)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] == 4:
            landmarks = data[:, 1:4]
        elif data.shape[1] >= 3:
            landmarks = data[:, :3]
        else:
            raise ValueError("Format inconnu (moins de 3 colonnes par ligne).")
        return landmarks
    except Exception as e:
        st.error(f"Erreur lors du parsing du fichier .bnd: {e}")
        return None

def find_nearest_vertex(vertices, point):
    diff = vertices - point
    dist = np.linalg.norm(diff, axis=1)
    return np.argmin(dist)

def plot_mesh_and_landmarks(mesh, landmarks, left_index=None, right_index=None):
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.5,
        color='lightblue',
        name='Maillage'
    ))
    fig.add_trace(go.Scatter3d(
        x=landmarks[:, 0],
        y=landmarks[:, 1],
        z=landmarks[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Landmarks'
    ))
    if left_index is not None and left_index < landmarks.shape[0]:
        fig.add_trace(go.Scatter3d(
            x=[landmarks[left_index, 0]],
            y=[landmarks[left_index, 1]],
            z=[landmarks[left_index, 2]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name=f'Œil intérieur gauche (index {left_index})'
        ))
    if right_index is not None and right_index < landmarks.shape[0]:
        fig.add_trace(go.Scatter3d(
            x=[landmarks[right_index, 0]],
            y=[landmarks[right_index, 1]],
            z=[landmarks[right_index, 2]],
            mode='markers',
            marker=dict(size=10, color='orange'),
            name=f'Œil intérieur droit (index {right_index})'
        ))
    fig.update_layout(title="Maillage et repères", scene=dict(aspectmode='data'))
    return fig

def plot_contours(contours_list, target_list, title="Contours"):
    """
    Affiche les contours dans une figure Plotly.
    Pour le type "diff", chaque contour est attendu sous forme de tuple (left, right).
    """
    fig = go.Figure()
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'magenta', 'brown', 'cyan']
    for i, contour in enumerate(contours_list):
        color = colors[i % len(colors)]
        if isinstance(contour, (list, tuple)) and len(contour) == 2:
            left_curve, right_curve = contour
            fig.add_trace(go.Scatter3d(
                x=left_curve[:, 0],
                y=left_curve[:, 1],
                z=left_curve[:, 2],
                mode='markers',
                marker=dict(size=3, color=color),
                name=f'Diff branche gauche C={target_list[i]:.2f}'
            ))
            fig.add_trace(go.Scatter3d(
                x=right_curve[:, 0],
                y=right_curve[:, 1],
                z=right_curve[:, 2],
                mode='markers',
                marker=dict(size=3, color=color),
                name=f'Diff branche droite C={target_list[i]:.2f}'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=contour[:, 0],
                y=contour[:, 1],
                z=contour[:, 2],
                mode='markers',
                marker=dict(size=3, color=color),
                name=f'Contour C={target_list[i]:.2f}'
            ))
    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    return fig

#######################################
# Fonctions pour le calcul géodésique (représentation bipolaire)
#######################################
def build_sparse_adjacency(vertices, faces):
    num_vertices = len(vertices)
    adjacency_matrix = lil_matrix((num_vertices, num_vertices), dtype=np.float32)
    for face in faces:
        for i in range(3):
            for j in range(i+1, 3):
                vi, vj = face[i], face[j]
                dist = np.linalg.norm(vertices[vi] - vertices[vj])
                adjacency_matrix[vi, vj] = dist
                adjacency_matrix[vj, vi] = dist
    return adjacency_matrix.tocsr()

def fast_dijkstra_sparse(adjacency_matrix, start_vertex):
    num_vertices = adjacency_matrix.shape[0]
    distances = np.full(num_vertices, np.inf)
    distances[start_vertex] = 0
    pq = [(0, start_vertex)]
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, edge_weight in zip(adjacency_matrix[current_vertex].indices,
                                         adjacency_matrix[current_vertex].data):
            new_distance = current_distance + edge_weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))
    return torch.tensor(distances, dtype=torch.float32, device=device)

def compute_bipolar_representation(vertices, faces, ref_idx1, ref_idx2):
    st.write("[INFO] Calcul des potentiels géodésiques avec CUDA...")
    adj = build_sparse_adjacency(vertices, faces)
    d1 = fast_dijkstra_sparse(adj, ref_idx1).to(device)
    d2 = fast_dijkstra_sparse(adj, ref_idx2).to(device)
    somme = d1 + d2
    diff = torch.abs(d1 - d2)
    st.write(f"[INFO] Plage somme: {torch.min(somme).item()} à {torch.max(somme).item()}")
    st.write(f"[INFO] Plage diff: {torch.min(diff).item()} à {torch.max(diff).item()}")
    return somme.cpu(), diff.cpu()

#######################################
# Extraction des contours et ICP sur ces contours
#######################################
def extract_contour(mesh, potentials, target_value, tol=0.5):
    indices = np.where(np.abs(potentials - target_value) < tol)[0]
    st.write(f"[INFO] Extraction du contour: {len(indices)} points trouvés pour cible {target_value} avec tolérance {tol}")
    return mesh.vertices[indices]

def icp_registration_pointcloud(source_points, target_points, threshold=0.5, max_iter=50):
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    trans_init = np.eye(4, dtype=np.float64)
    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return reg_result.transformation, reg_result, source_pcd, target_pcd

def plot_icp_error_histogram(source_pcd, target_pcd):
    distances = target_pcd.compute_point_cloud_distance(source_pcd)
    distances = np.asarray(distances)
    mean_error = np.mean(distances)
    rmse = np.sqrt(np.mean(distances**2))
    st.write(f"[INFO] Erreur moyenne: {mean_error:.4f}")
    st.write(f"[INFO] RMSE: {rmse:.4f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(distances, bins=30, color='gray', edgecolor='black')
    ax.set_title("Histogramme des erreurs ICP")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Nombre de points")
    st.pyplot(fig)
    return mean_error, rmse

#######################################
# Fonctions pour l'ordonnancement et interpolation B-spline
#######################################
def order_contour_points(points, contour_type="sum"):
    """
    Pour le level set "sum", on utilise un NN-ordering pour obtenir une courbe unique.
    """
    if len(points) <= 1:
        return points.copy()
    
    if contour_type == "sum":
        dist_matrix = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
        i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        ordered = []
        remaining = set(range(len(points)))
        current = i
        ordered.append(current)
        remaining.remove(current)
        while remaining:
            nearest = min(remaining, key=lambda x: np.linalg.norm(points[current] - points[x]))
            ordered.append(nearest)
            remaining.remove(nearest)
            current = nearest
        if ordered[-1] != j:
            ordered = ordered[::-1]
        return points[ordered]
    else:
        return points  # Ne devrait pas être appelé pour "diff"

def order_diff_contour_points(points):
    """
    Divise les points du level set "diff" en deux branches (gauche et droite) 
    selon la médiane des x, et les ordonne par rapport à la coordonnée y.
    """
    centroid = np.mean(points, axis=0)
    left_mask = points[:, 0] < centroid[0]
    left_points = points[left_mask]
    right_points = points[~left_mask]
    if len(left_points) > 0:
        left_ordered = left_points[np.argsort(left_points[:, 1])]
    else:
        left_ordered = np.empty((0, 3))
    if len(right_points) > 0:
        right_ordered = right_points[np.argsort(right_points[:, 1])]
    else:
        right_ordered = np.empty((0, 3))
    return left_ordered, right_ordered

def fit_bspline_curve(points, n_samples=50, s=0, contour_type="sum"):
    """
    Pour "sum", ajuste une B-spline sur la courbe ordonnée.
    """
    if len(points) < 4:
        return points, None, None
    points_ordered = order_contour_points(points, contour_type=contour_type)
    # Vérifier si le contour est fermé (pour "sum")
    is_closed = (contour_type == "sum") and (np.linalg.norm(points_ordered[0] - points_ordered[-1]) < 1e-3)
    k = 3 if len(points_ordered) >= 4 else min(3, len(points_ordered)-1)
    try:
        dists = np.linalg.norm(np.diff(points_ordered, axis=0), axis=1)
        arc_length = np.concatenate(([0], np.cumsum(dists)))
        arc_length_norm = arc_length / arc_length[-1] if arc_length[-1] != 0 else np.linspace(0, 1, len(points_ordered))
        tck, u = splprep(points_ordered.T, u=arc_length_norm, s=s, k=k, per=is_closed)
        u_new = np.linspace(0, 1, n_samples)
        if is_closed:
            u_new = u_new[:-1]  # éviter la duplication
        curve = np.array(splev(u_new, tck)).T
        return curve, tck, u_new
    except Exception as e:
        st.error(f"B-spline error: {str(e)}")
        return points_ordered, None, None

def fit_bspline_diff_curve(points, n_samples=50, s=0):
    """
    Pour le level set "diff", sépare d'abord en deux branches puis ajuste une B-spline pour chacune.
    Retourne un tuple (left_curve, right_curve).
    """
    left_ordered, right_ordered = order_diff_contour_points(points)
    left_curve = left_ordered
    right_curve = right_ordered
    if len(left_ordered) >= 4:
        try:
            dists = np.linalg.norm(np.diff(left_ordered, axis=0), axis=1)
            arc_length = np.concatenate(([0], np.cumsum(dists)))
            arc_length_norm = arc_length / arc_length[-1] if arc_length[-1] != 0 else np.linspace(0, 1, len(left_ordered))
            tck_left, _ = splprep(left_ordered.T, u=arc_length_norm, s=s, k=3, per=False)
            u_new = np.linspace(0, 1, n_samples)
            left_curve = np.array(splev(u_new, tck_left)).T
        except Exception as e:
            st.error(f"B-spline error (left branch): {str(e)}")
    if len(right_ordered) >= 4:
        try:
            dists = np.linalg.norm(np.diff(right_ordered, axis=0), axis=1)
            arc_length = np.concatenate(([0], np.cumsum(dists)))
            arc_length_norm = arc_length / arc_length[-1] if arc_length[-1] != 0 else np.linspace(0, 1, len(right_ordered))
            tck_right, _ = splprep(right_ordered.T, u=arc_length_norm, s=s, k=3, per=False)
            u_new = np.linspace(0, 1, n_samples)
            right_curve = np.array(splev(u_new, tck_right)).T
        except Exception as e:
            st.error(f"B-spline error (right branch): {str(e)}")
    return left_curve, right_curve

def process_contours(contours, contour_type, n_samples, s):
    """
    Pour chaque contour extrait, applique l'interpolation par B-spline.
    Pour "sum", renvoie une liste d'arrays.
    Pour "diff", renvoie une liste de tuples (left_curve, right_curve).
    """
    processed = []
    if contour_type == "sum":
        for contour in contours:
            if len(contour) < 4:
                processed.append(contour)
            else:
                curve, _, _ = fit_bspline_curve(contour, n_samples=n_samples, s=s, contour_type=contour_type)
                processed.append(curve if curve is not None else contour)
    elif contour_type == "diff":
        for contour in contours:
            if len(contour) < 4:
                # Dupliquer si insuffisant
                processed.append((contour, contour))
            else:
                left_curve, right_curve = fit_bspline_diff_curve(contour, n_samples=n_samples, s=s)
                processed.append((left_curve, right_curve))
    return processed

#######################################
# Interface Streamlit - Chargement et paramètres
#######################################
st.sidebar.header("Chargement des fichiers")
uploaded_source_mesh = st.sidebar.file_uploader("Maillage source (OBJ)", type=["obj"])
uploaded_source_bnd = st.sidebar.file_uploader("Repères source (.bnd)", type=["bnd"])
uploaded_target_mesh = st.sidebar.file_uploader("Maillage cible (OBJ)", type=["obj"])
uploaded_target_bnd = st.sidebar.file_uploader("Repères cible (.bnd)", type=["bnd"])

st.sidebar.header("Indices des repères (pour les yeux intérieurs)")
source_ref_left = st.sidebar.number_input("Indice repère œil intérieur gauche (source)", value=0, step=1)
source_ref_right = st.sidebar.number_input("Indice repère œil intérieur droit (source)", value=8, step=1)
target_ref_left = st.sidebar.number_input("Indice repère œil intérieur gauche (cible)", value=0, step=1)
target_ref_right = st.sidebar.number_input("Indice repère œil intérieur droit (cible)", value=8, step=1)

st.sidebar.header("Paramètres ICP et extraction")
icp_threshold = st.sidebar.slider("Seuil ICP", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
icp_max_iter = st.sidebar.slider("Nombre max d'itérations ICP", min_value=10, max_value=200, value=50, step=10)
contour_tol = st.sidebar.slider("Tolérance pour extraction de contour", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

st.sidebar.header("Distances cibles")
source_sum_str = st.sidebar.text_input("Target distances (sum) pour la source", 
                                         value="97,108,119,130,141,152,163,174,185,196,207,218,229")
source_diff_str = st.sidebar.text_input("Target distances (diff) pour la source", 
                                          value="18,19,20,21,22,23,24,25,26,27,28,29,30")
target_sum_str = st.sidebar.text_input("Target distances (sum) pour la cible", 
                                         value="97,108,119,130,141,152,163,174,185,196,207,218,229")
target_diff_str = st.sidebar.text_input("Target distances (diff) pour la cible", 
                                          value="18,19,20,21,22,23,24,25,26,27,28,29,30")

# Paramètres pour B-spline
st.sidebar.header("Paramètres B-spline")
n_samples = st.sidebar.slider("Nombre de points pour B-spline", min_value=10, max_value=200, value=50, step=5)
spline_s = st.sidebar.slider("Smoothing (s) pour B-spline", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

# Mode d'alignement : Individuel ou Batch
align_mode = st.sidebar.radio("Mode d'alignement", ("Paire unique (sélectionnée)", "Batch ICP (tous les contours)"))

# Pour le mode individuel, sélection de la paire pour "sum"
if align_mode == "Paire unique (sélectionnée)":
    selected_pair_sum = st.sidebar.selectbox("Sélectionner la paire de contours 'sum' pour ICP (index)", 
                                               list(range(min(len(source_sum_str.split(',')), len(target_sum_str.split(','))))))
    # Pour "diff", on ajoute une sélection de paire ainsi que le choix de branche (left/right)
    selected_pair_diff = st.sidebar.selectbox("Sélectionner la paire de contours 'diff' pour ICP (index)", 
                                                list(range(min(len(source_diff_str.split(',')), len(target_diff_str.split(','))))))
    selected_diff_branch = st.sidebar.radio("Sélectionner la branche pour 'diff'", ("left", "right"))

try:
    target_distances_source_sum = [float(x.strip()) for x in source_sum_str.split(',') if x.strip() != ""]
except Exception as e:
    st.error(f"Erreur de parsing des distances cibles source (sum): {e}")
    target_distances_source_sum = []
try:
    target_distances_source_diff = [float(x.strip()) for x in source_diff_str.split(',') if x.strip() != ""]
except Exception as e:
    st.error(f"Erreur de parsing des distances cibles source (diff): {e}")
    target_distances_source_diff = []
try:
    target_distances_target_sum = [float(x.strip()) for x in target_sum_str.split(',') if x.strip() != ""]
except Exception as e:
    st.error(f"Erreur de parsing des distances cibles cible (sum): {e}")
    target_distances_target_sum = []
try:
    target_distances_target_diff = [float(x.strip()) for x in target_diff_str.split(',') if x.strip() != ""]
except Exception as e:
    st.error(f"Erreur de parsing des distances cibles cible (diff): {e}")
    target_distances_target_diff = []

if st.sidebar.button("Exécuter l'alignement sur les contours"):
    if uploaded_source_mesh and uploaded_source_bnd and uploaded_target_mesh and uploaded_target_bnd:
        # Sauvegarder temporairement les fichiers uploadés
        with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp:
            tmp.write(uploaded_source_mesh.getvalue())
            source_mesh_path = tmp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bnd") as tmp:
            tmp.write(uploaded_source_bnd.getvalue())
            source_bnd_path = tmp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp:
            tmp.write(uploaded_target_mesh.getvalue())
            target_mesh_path = tmp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bnd") as tmp:
            tmp.write(uploaded_target_bnd.getvalue())
            target_bnd_path = tmp.name

        st.write("[INFO] Chargement des maillages...")
        mesh_source = trimesh.load(source_mesh_path, process=True)
        vertices_source = np.array(mesh_source.vertices)
        faces_source = np.array(mesh_source.faces)
        mesh_target = trimesh.load(target_mesh_path, process=True)
        vertices_target = np.array(mesh_target.vertices)
        faces_target = np.array(mesh_target.faces)
        
        st.write("[INFO] Lecture des repères...")
        landmarks_source = read_bnd_file(open(source_bnd_path, "rb").read())
        landmarks_target = read_bnd_file(open(target_bnd_path, "rb").read())
        if landmarks_source is None or landmarks_target is None:
            st.error("Erreur lors de la lecture des fichiers .bnd")
        else:
            st.write("[INFO] 5 premières lignes des repères source :")
            st.dataframe(landmarks_source[:5])
            st.write("[INFO] 5 premières lignes des repères cible :")
            st.dataframe(landmarks_target[:5])
            
            # Extraire les repères pour les yeux intérieurs
            ref_source_left = landmarks_source[int(source_ref_left)]
            ref_source_right = landmarks_source[int(source_ref_right)]
            ref_target_left = landmarks_target[int(target_ref_left)]
            ref_target_right = landmarks_target[int(target_ref_right)]
            
            # Trouver les indices des vertex les plus proches des repères
            idx_source_left = find_nearest_vertex(vertices_source, ref_source_left)
            idx_source_right = find_nearest_vertex(vertices_source, ref_source_right)
            idx_target_left = find_nearest_vertex(vertices_target, ref_target_left)
            idx_target_right = find_nearest_vertex(vertices_target, ref_target_right)
            
            st.write(f"[INFO] Indices repères source: gauche={idx_source_left}, droite={idx_source_right}")
            st.write(f"[INFO] Indices repères cible: gauche={idx_target_left}, droite={idx_target_right}")
            
            # Calculer la représentation bipolaire pour chaque maillage
            somme_source, diff_source = compute_bipolar_representation(vertices_source, faces_source, idx_source_left, idx_source_right)
            somme_target, diff_target = compute_bipolar_representation(vertices_target, faces_target, idx_target_left, idx_target_right)
            
            # Extraction des contours pour "sum" et "diff"
            contours_source_sum = [extract_contour(mesh_source, somme_source.numpy(), c, tol=contour_tol) for c in target_distances_source_sum]
            contours_source_diff = [extract_contour(mesh_source, diff_source.numpy(), c, tol=contour_tol) for c in target_distances_source_diff]
            contours_target_sum = [extract_contour(mesh_target, somme_target.numpy(), c, tol=contour_tol) for c in target_distances_target_sum]
            contours_target_diff = [extract_contour(mesh_target, diff_target.numpy(), c, tol=contour_tol) for c in target_distances_target_diff]
            
            # Appliquer l'ordonnancement et l'interpolation B-spline
            contours_source_sum_bspline = process_contours(contours_source_sum, "sum", n_samples, spline_s)
            contours_target_sum_bspline = process_contours(contours_target_sum, "sum", n_samples, spline_s)
            contours_source_diff_bspline = process_contours(contours_source_diff, "diff", n_samples, spline_s)
            contours_target_diff_bspline = process_contours(contours_target_diff, "diff", n_samples, spline_s)
            
            # Visualisation des contours interpolés
            st.write("[INFO] Visualisation des contours 'sum' pour la source (B-spline)")
            fig_source_sum = plot_contours(contours_source_sum_bspline, target_distances_source_sum, title="Contours source (sum)")
            st.plotly_chart(fig_source_sum, use_container_width=True)
            st.write("[INFO] Visualisation des contours 'diff' pour la source (B-spline)")
            fig_source_diff = plot_contours(contours_source_diff_bspline, target_distances_source_diff, title="Contours source (diff)")
            st.plotly_chart(fig_source_diff, use_container_width=True)
            st.write("[INFO] Visualisation des contours 'sum' pour la cible (B-spline)")
            fig_target_sum = plot_contours(contours_target_sum_bspline, target_distances_target_sum, title="Contours cible (sum)")
            st.plotly_chart(fig_target_sum, use_container_width=True)
            st.write("[INFO] Visualisation des contours 'diff' pour la cible (B-spline)")
            fig_target_diff = plot_contours(contours_target_diff_bspline, target_distances_target_diff, title="Contours cible (diff)")
            st.plotly_chart(fig_target_diff, use_container_width=True)
            
            # Calcul de la translation initiale basée sur les repères
            mean_source = (ref_source_left + ref_source_right) / 2.0
            mean_target = (ref_target_left + ref_target_right) / 2.0
            initial_translation = mean_target - mean_source
            st.write("[INFO] Translation initiale basée sur les repères:", initial_translation)
            
            if align_mode == "Paire unique (sélectionnée)":
                # Sélection pour "sum"
                st.write(f"[INFO] Utilisation de la paire de contours 'sum' d'indice {selected_pair_sum} pour ICP")
                selected_contour_source_sum = contours_source_sum_bspline[selected_pair_sum]
                selected_contour_target_sum = contours_target_sum_bspline[selected_pair_sum]
                
                # Pour "diff", sélection selon la branche choisie
                st.write(f"[INFO] Utilisation de la paire de contours 'diff' d'indice {selected_pair_diff} pour ICP, branche {selected_diff_branch}")
                diff_source_tuple = contours_source_diff_bspline[selected_pair_diff]  # (left, right)
                diff_target_tuple = contours_target_diff_bspline[selected_pair_diff]
                if selected_diff_branch == "left":
                    selected_contour_source_diff = diff_source_tuple[0]
                    selected_contour_target_diff = diff_target_tuple[0]
                else:
                    selected_contour_source_diff = diff_source_tuple[1]
                    selected_contour_target_diff = diff_target_tuple[1]
                
                # Visualisation pré-alignement
                pre_align_fig_sum = plot_contours([selected_contour_source_sum, selected_contour_target_sum], 
                                                   [target_distances_source_sum[selected_pair_sum], target_distances_target_sum[selected_pair_sum]],
                                                   title="Contours 'sum' sélectionnés avant ICP")
                st.plotly_chart(pre_align_fig_sum, use_container_width=True)
                pre_align_fig_diff = plot_contours([selected_contour_source_diff, selected_contour_target_diff], 
                                                    [target_distances_source_diff[selected_pair_diff], target_distances_target_diff[selected_pair_diff]],
                                                    title="Contours 'diff' sélectionnés avant ICP")
                st.plotly_chart(pre_align_fig_diff, use_container_width=True)
                
                # Appliquer la translation initiale aux contours sélectionnés
                src_sum_aligned_init = selected_contour_source_sum + initial_translation
                src_diff_aligned_init = selected_contour_source_diff + initial_translation
                
                # ICP pour "sum"
                transformation_icp_sum, reg_result_sum, src_pcd_sum, tgt_pcd_sum = icp_registration_pointcloud(
                    src_sum_aligned_init, selected_contour_target_sum, threshold=icp_threshold, max_iter=icp_max_iter)
                st.write("Matrice de transformation ICP finale (sum):")
                st.write(transformation_icp_sum)
                st.write(f"Fitness (sum): {reg_result_sum.fitness:.4f}")
                st.write(f"Inlier RMSE (sum): {reg_result_sum.inlier_rmse:.4f}")
                
                # ICP pour "diff"
                transformation_icp_diff, reg_result_diff, src_pcd_diff, tgt_pcd_diff = icp_registration_pointcloud(
                    src_diff_aligned_init, selected_contour_target_diff, threshold=icp_threshold, max_iter=icp_max_iter)
                st.write("Matrice de transformation ICP finale (diff):")
                st.write(transformation_icp_diff)
                st.write(f"Fitness (diff): {reg_result_diff.fitness:.4f}")
                st.write(f"Inlier RMSE (diff): {reg_result_diff.inlier_rmse:.4f}")
                
                # Visualisation post-alignement pour "sum"
                src_points_aligned_sum = np.asarray(src_pcd_sum.points)
                post_align_fig_sum = go.Figure()
                post_align_fig_sum.add_trace(go.Scatter3d(
                    x=src_points_aligned_sum[:, 0],
                    y=src_points_aligned_sum[:, 1],
                    z=src_points_aligned_sum[:, 2],
                    mode='markers',
                    marker=dict(size=3, color='red'),
                    name='Contour source (sum) après ICP'
                ))
                post_align_fig_sum.add_trace(go.Scatter3d(
                    x=selected_contour_target_sum[:, 0],
                    y=selected_contour_target_sum[:, 1],
                    z=selected_contour_target_sum[:, 2],
                    mode='markers',
                    marker=dict(size=3, color='blue'),
                    name='Contour cible (sum)'
                ))
                post_align_fig_sum.update_layout(title="Contours 'sum' après ICP", scene=dict(aspectmode='data'))
                st.plotly_chart(post_align_fig_sum, use_container_width=True)
                
                # Visualisation post-alignement pour "diff"
                src_points_aligned_diff = np.asarray(src_pcd_diff.points)
                post_align_fig_diff = go.Figure()
                post_align_fig_diff.add_trace(go.Scatter3d(
                    x=src_points_aligned_diff[:, 0],
                    y=src_points_aligned_diff[:, 1],
                    z=src_points_aligned_diff[:, 2],
                    mode='markers',
                    marker=dict(size=3, color='red'),
                    name='Contour source (diff) après ICP'
                ))
                post_align_fig_diff.add_trace(go.Scatter3d(
                    x=selected_contour_target_diff[:, 0],
                    y=selected_contour_target_diff[:, 1],
                    z=selected_contour_target_diff[:, 2],
                    mode='markers',
                    marker=dict(size=3, color='blue'),
                    name='Contour cible (diff)'
                ))
                post_align_fig_diff.update_layout(title="Contours 'diff' après ICP", scene=dict(aspectmode='data'))
                st.plotly_chart(post_align_fig_diff, use_container_width=True)
                
                # Histogrammes des erreurs ICP
                st.write("[INFO] Erreurs ICP pour le type 'sum':")
                mean_error_sum, rmse_sum = plot_icp_error_histogram(src_pcd_sum, tgt_pcd_sum)
                st.write(f"[INFO] RMSE (sum): {rmse_sum:.4f}")
                
                st.write("[INFO] Erreurs ICP pour le type 'diff':")
                mean_error_diff, rmse_diff = plot_icp_error_histogram(src_pcd_diff, tgt_pcd_diff)
                st.write(f"[INFO] RMSE (diff): {rmse_diff:.4f}")
                
            else:  # Batch ICP mode
                st.write("[INFO] Exécution du Batch ICP sur tous les contours (B-spline appliquée)")
                
                # Pour "sum"
                aligned_contours_sum = []
                metrics_sum = []
                for i in range(len(contours_source_sum_bspline)):
                    source_curve = contours_source_sum_bspline[i] + initial_translation
                    target_curve = contours_target_sum_bspline[i]
                    T, reg_result, src_pcd, tgt_pcd = icp_registration_pointcloud(source_curve, target_curve, threshold=icp_threshold, max_iter=icp_max_iter)
                    aligned_contours_sum.append(np.asarray(src_pcd.points))
                    metrics_sum.append((reg_result.fitness, reg_result.inlier_rmse))
                
                # Pour "diff", traiter séparément les branches gauche et droite
                aligned_diff_left = []
                aligned_diff_right = []
                metrics_diff_left = []
                metrics_diff_right = []
                for i in range(len(contours_source_diff_bspline)):
                    src_left, src_right = contours_source_diff_bspline[i]
                    tgt_left, tgt_right = contours_target_diff_bspline[i]
                    # ICP pour la branche gauche
                    T_left, reg_result_left, src_pcd_left, tgt_pcd_left = icp_registration_pointcloud(src_left + initial_translation, tgt_left, threshold=icp_threshold, max_iter=icp_max_iter)
                    aligned_diff_left.append(np.asarray(src_pcd_left.points))
                    metrics_diff_left.append((reg_result_left.fitness, reg_result_left.inlier_rmse))
                    # ICP pour la branche droite
                    T_right, reg_result_right, src_pcd_right, tgt_pcd_right = icp_registration_pointcloud(src_right + initial_translation, tgt_right, threshold=icp_threshold, max_iter=icp_max_iter)
                    aligned_diff_right.append(np.asarray(src_pcd_right.points))
                    metrics_diff_right.append((reg_result_right.fitness, reg_result_right.inlier_rmse))
                
                # Visualisation agrégée pour "sum"
                batch_fig_sum = go.Figure()
                for i, curve in enumerate(aligned_contours_sum):
                    batch_fig_sum.add_trace(go.Scatter3d(
                        x=curve[:, 0],
                        y=curve[:, 1],
                        z=curve[:, 2],
                        mode='markers',
                        marker=dict(size=3, color='red'),
                        name=f'Source ICP {i}'
                    ))
                    batch_fig_sum.add_trace(go.Scatter3d(
                        x=contours_target_sum_bspline[i][:, 0],
                        y=contours_target_sum_bspline[i][:, 1],
                        z=contours_target_sum_bspline[i][:, 2],
                        mode='markers',
                        marker=dict(size=3, color='blue'),
                        name=f'Cible {i}'
                    ))
                batch_fig_sum.update_layout(title="Batch ICP - Contours 'sum' après alignement", scene=dict(aspectmode='data'))
                st.plotly_chart(batch_fig_sum, use_container_width=True)
                
                # Visualisation agrégée pour "diff" (affichage séparé pour gauche et droite)
                batch_fig_diff = go.Figure()
                for i, curve in enumerate(aligned_diff_left):
                    batch_fig_diff.add_trace(go.Scatter3d(
                        x=curve[:, 0],
                        y=curve[:, 1],
                        z=curve[:, 2],
                        mode='markers',
                        marker=dict(size=3, color='red'),
                        name=f'Source ICP diff gauche {i}'
                    ))
                    batch_fig_diff.add_trace(go.Scatter3d(
                        x=contours_target_diff_bspline[i][0][:, 0],
                        y=contours_target_diff_bspline[i][0][:, 1],
                        z=contours_target_diff_bspline[i][0][:, 2],
                        mode='markers',
                        marker=dict(size=3, color='blue'),
                        name=f'Cible diff gauche {i}'
                    ))
                for i, curve in enumerate(aligned_diff_right):
                    batch_fig_diff.add_trace(go.Scatter3d(
                        x=curve[:, 0],
                        y=curve[:, 1],
                        z=curve[:, 2],
                        mode='markers',
                        marker=dict(size=3, color='orange'),
                        name=f'Source ICP diff droite {i}'
                    ))
                    batch_fig_diff.add_trace(go.Scatter3d(
                        x=contours_target_diff_bspline[i][1][:, 0],
                        y=contours_target_diff_bspline[i][1][:, 1],
                        z=contours_target_diff_bspline[i][1][:, 2],
                        mode='markers',
                        marker=dict(size=3, color='green'),
                        name=f'Cible diff droite {i}'
                    ))
                batch_fig_diff.update_layout(title="Batch ICP - Contours 'diff' après alignement", scene=dict(aspectmode='data'))
                st.plotly_chart(batch_fig_diff, use_container_width=True)
                
                # Affichage des métriques moyennes
                avg_fitness_sum = np.mean([m[0] for m in metrics_sum])
                avg_rmse_sum = np.mean([m[1] for m in metrics_sum])
                avg_fitness_diff_left = np.mean([m[0] for m in metrics_diff_left])
                avg_rmse_diff_left = np.mean([m[1] for m in metrics_diff_left])
                avg_fitness_diff_right = np.mean([m[0] for m in metrics_diff_right])
                avg_rmse_diff_right = np.mean([m[1] for m in metrics_diff_right])
                st.write(f"[INFO] Moyenne des fitness (sum): {avg_fitness_sum:.4f} et RMSE: {avg_rmse_sum:.4f}")
                st.write(f"[INFO] Moyenne des fitness (diff gauche): {avg_fitness_diff_left:.4f} et RMSE: {avg_rmse_diff_left:.4f}")
                st.write(f"[INFO] Moyenne des fitness (diff droite): {avg_fitness_diff_right:.4f} et RMSE: {avg_rmse_diff_right:.4f}")
            
            st.markdown("""
            **Prochaines étapes expérimentales suggérées :**
            1. Varier les distances cibles et la tolérance pour l'extraction des contours et observer l'impact sur l'alignement ICP.
            2. Effectuer une PCA sur le système de coordonnées intrinsèque S(U,V) extrait des potentiels.
            3. Comparer l'approche bipolaire (sum et diff) avec une approche à trois pôles pour déterminer la méthode la plus générale et plus robuste.
            4. Envisager l'entraînement d'un modèle CNN pour la reconnaissance d'expressions basé sur ces représentations.
            """)
    else:
        st.error("Veuillez uploader les fichiers maillage (.obj) et repères (.bnd) pour la source et la cible.")
