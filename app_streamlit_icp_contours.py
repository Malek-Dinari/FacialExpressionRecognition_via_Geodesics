#!/usr/bin/env python
"""
app_streamlit_icp_contours.py

Cette application Streamlit permet de :
  - Charger deux maillages 3D (source et cible) et leurs repères (.bnd).
  - Extraire les repères correspondant aux bords intérieurs des yeux (via indices fournis ou par défaut).
  - Calculer la représentation bipolaire (potentiels géodésiques) en utilisant ces repères.
  - Extraire des contours (level sets) sur le maillage via des distances cibles (entrées manuelles) et une tolérance.
  - Visualiser tous les contours extraits avant alignement.
  - Sélectionner une paire de contours pour ICP, pour les "sum" et pour les "diff".
  - Initialiser l'ICP par translation (basée sur les repères) et aligner les contours sélectionnés.
  - Visualiser les contours avant et après ICP.
  - Afficher la matrice de transformation ICP, les métriques (fitness, inlier RMSE) et un histogramme des erreurs.

Quelques notions importantes :
  - **ICP (Iterative Closest Point)** recherche la transformation (translation et rotation) minimisant la distance entre deux ensembles de points.
  - **Mesh matching** utilise des caractéristiques (ici, les repères extraits et la représentation bipolaire) pour aligner deux maillages.
  - La **représentation bipolaire** est définie par les potentiels géodésiques calculés depuis deux repères (les yeux intérieurs). Les level sets sont les ensembles de points où la somme ou la différence des potentiels est constante.

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
    """
    Lire un fichier .bnd et retourner les repères (landmarks) sous forme d'un tableau (N, 3).

    Pour BU3D-FE, le fichier est en format texte.
    Exemple de ligne :
      7323\t\t-39.6402\t\t47.0837\t\t95.2473
    La première colonne est un index, les trois suivantes sont X, Y, Z.
    """
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
    """
    Trouver l'indice du vertex le plus proche du point donné.
    """
    diff = vertices - point
    dist = np.linalg.norm(diff, axis=1)
    return np.argmin(dist)

def plot_mesh_and_landmarks(mesh, landmarks, left_index=None, right_index=None):
    """
    Afficher le maillage et les repères sur un graphique 3D interactif avec Plotly.
    """
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
    Afficher plusieurs ensembles de points (nuages) représentant les contours.
    Chaque contour est tracé avec une couleur différente.
    """
    fig = go.Figure()
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'magenta', 'brown', 'cyan']
    for i, contour in enumerate(contours_list):
        color = colors[i % len(colors)]
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
# Interface Streamlit
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

# Entrées de distances cibles sous forme de liste pour "sum" et "diff"
source_sum_str = st.sidebar.text_input("Target distances (sum) pour la source", 
                                         value="97,108,119,130,141,152,163,174,185,196,207,218,229")
source_diff_str = st.sidebar.text_input("Target distances (diff) pour la source", 
                                          value="18,19,20,21,22,23,24,25,26,27,28,29,30")
target_sum_str = st.sidebar.text_input("Target distances (sum) pour la cible", 
                                         value="97,108,119,130,141,152,163,174,185,196,207,218,229")
target_diff_str = st.sidebar.text_input("Target distances (diff) pour la cible", 
                                          value="18,19,20,21,22,23,24,25,26,27,28,29,30")

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

# Sélection des paires de contours à aligner pour chaque type
selected_pair_sum = st.sidebar.selectbox("Sélectionner la paire de contours (sum) pour ICP (index)", 
                                           list(range(min(len(target_distances_source_sum), len(target_distances_target_sum)))) if target_distances_source_sum and target_distances_target_sum else [0])
selected_pair_diff = st.sidebar.selectbox("Sélectionner la paire de contours (diff) pour ICP (index)", 
                                            list(range(min(len(target_distances_source_diff), len(target_distances_target_diff)))) if target_distances_source_diff and target_distances_target_diff else [0])

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
            
            # Extraire les contours pour "sum" et "diff" pour la source
            contours_source_sum = []
            for c in target_distances_source_sum:
                cs = extract_contour(mesh_source, somme_source.numpy(), c, tol=contour_tol)
                contours_source_sum.append(cs)
            contours_source_diff = []
            for c in target_distances_source_diff:
                cd = extract_contour(mesh_source, diff_source.numpy(), c, tol=contour_tol)
                contours_source_diff.append(cd)
            
            # Extraire les contours pour "sum" et "diff" pour la cible
            contours_target_sum = []
            for c in target_distances_target_sum:
                ct = extract_contour(mesh_target, somme_target.numpy(), c, tol=contour_tol)
                contours_target_sum.append(ct)
            contours_target_diff = []
            for c in target_distances_target_diff:
                cd = extract_contour(mesh_target, diff_target.numpy(), c, tol=contour_tol)
                contours_target_diff.append(cd)
            
            # Visualiser les contours extraits pour la source et la cible
            st.write("[INFO] Visualisation des contours 'sum' pour la source")
            fig_source_sum = plot_contours(contours_source_sum, target_distances_source_sum, title="Contours source (sum)")
            st.plotly_chart(fig_source_sum, use_container_width=True)
            st.write("[INFO] Visualisation des contours 'diff' pour la source")
            fig_source_diff = plot_contours(contours_source_diff, target_distances_source_diff, title="Contours source (diff)")
            st.plotly_chart(fig_source_diff, use_container_width=True)
            st.write("[INFO] Visualisation des contours 'sum' pour la cible")
            fig_target_sum = plot_contours(contours_target_sum, target_distances_target_sum, title="Contours cible (sum)")
            st.plotly_chart(fig_target_sum, use_container_width=True)
            st.write("[INFO] Visualisation des contours 'diff' pour la cible")
            fig_target_diff = plot_contours(contours_target_diff, target_distances_target_diff, title="Contours cible (diff)")
            st.plotly_chart(fig_target_diff, use_container_width=True)
            
            # Sélectionner la paire de contours pour ICP pour "sum" et pour "diff"
            selected_pair_sum = st.sidebar.selectbox("Sélectionner la paire de contours 'sum' pour ICP (index)", 
                                                       list(range(min(len(target_distances_source_sum), len(target_distances_target_sum)))) if target_distances_source_sum and target_distances_target_sum else [0])
            selected_pair_diff = st.sidebar.selectbox("Sélectionner la paire de contours 'diff' pour ICP (index)", 
                                                        list(range(min(len(target_distances_source_diff), len(target_distances_target_diff)))) if target_distances_source_diff and target_distances_target_diff else [0])
            
            st.write(f"[INFO] Utilisation de la paire de contours 'sum' d'indice {selected_pair_sum} pour ICP")
            selected_contour_source_sum = contours_source_sum[selected_pair_sum]
            selected_contour_target_sum = contours_target_sum[selected_pair_sum]
            
            st.write(f"[INFO] Utilisation de la paire de contours 'diff' d'indice {selected_pair_diff} pour ICP")
            selected_contour_source_diff = contours_source_diff[selected_pair_diff]
            selected_contour_target_diff = contours_target_diff[selected_pair_diff]
            
            # Visualisation pré-alignement
            pre_align_fig_sum = plot_contours([selected_contour_source_sum, selected_contour_target_sum], 
                                               [target_distances_source_sum[selected_pair_sum], target_distances_target_sum[selected_pair_sum]],
                                               title="Contours 'sum' sélectionnés avant ICP")
            st.plotly_chart(pre_align_fig_sum, use_container_width=True)
            pre_align_fig_diff = plot_contours([selected_contour_source_diff, selected_contour_target_diff], 
                                                [target_distances_source_diff[selected_pair_diff], target_distances_target_diff[selected_pair_diff]],
                                                title="Contours 'diff' sélectionnés avant ICP")
            st.plotly_chart(pre_align_fig_diff, use_container_width=True)
            
            # Calcul de la translation initiale (basée sur les repères)
            mean_source = (ref_source_left + ref_source_right) / 2.0
            mean_target = (ref_target_left + ref_target_right) / 2.0
            initial_translation = mean_target - mean_source
            st.write("[INFO] Translation initiale basée sur les repères:", initial_translation)
            
            # Appliquer la translation initiale aux contours sources sélectionnés
            selected_contour_source_sum_aligned = selected_contour_source_sum + initial_translation
            selected_contour_source_diff_aligned = selected_contour_source_diff + initial_translation
            
            # Exécuter l'ICP pour "sum"
            transformation_icp_sum, reg_result_sum, src_pcd_sum, tgt_pcd_sum = icp_registration_pointcloud(
                selected_contour_source_sum_aligned, selected_contour_target_sum, threshold=icp_threshold, max_iter=icp_max_iter)
            st.write("Matrice de transformation ICP finale (sum):")
            st.write(transformation_icp_sum)
            st.write(f"Fitness (sum): {reg_result_sum.fitness:.4f}")
            st.write(f"Inlier RMSE (sum): {reg_result_sum.inlier_rmse:.4f}")
            
            # Exécuter l'ICP pour "diff"
            transformation_icp_diff, reg_result_diff, src_pcd_diff, tgt_pcd_diff = icp_registration_pointcloud(
                selected_contour_source_diff_aligned, selected_contour_target_diff, threshold=icp_threshold, max_iter=icp_max_iter)
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
            
            # Tracer l'histogramme des erreurs ICP pour "sum" et "diff"
            st.write("[INFO] Erreurs ICP pour le type 'sum':")
            mean_error_sum, rmse_sum = plot_icp_error_histogram(src_pcd_sum, tgt_pcd_sum)
            st.write(f"[INFO] RMSE (sum): {rmse_sum:.4f}")
            
            st.write("[INFO] Erreurs ICP pour le type 'diff':")
            mean_error_diff, rmse_diff = plot_icp_error_histogram(src_pcd_diff, tgt_pcd_diff)
            st.write(f"[INFO] RMSE (diff): {rmse_diff:.4f}")
            
            st.markdown("""
            **Prochaines étapes expérimentales suggérées :**
            1. Varier les distances cibles et la tolérance pour l'extraction des contours et observer l'impact sur l'alignement ICP.
            2. Effectuer une PCA sur le système de coordonnées intrinsèque S(U,V) extrait des potentiels.
            3. Comparer l'approche bipolaire (sum et diff) avec une approche à trois pôles pour déterminer la méthode la plus robuste.
            4. Envisager l'entraînement d'un modèle CNN pour la reconnaissance d'expressions basé sur ces représentations.
            """)
    else:
        st.error("Veuillez uploader les fichiers maillage (.obj) et repères (.bnd) pour la source et la cible.")
