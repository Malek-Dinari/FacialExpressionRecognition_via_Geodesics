#!/usr/bin/env python
"""
app_streamlit_lda_bipolar.py

Cette application Streamlit permet de :
  - Charger un maillage 3D (face) et son fichier de repères (.bnd) du dataset BU3D-FE.
  - Extraire les repères correspondant aux bords intérieurs des yeux (indices 0 et 8 par défaut).
  - Calculer la représentation bipolaire (potentiels géodésiques) en utilisant ces repères.
  - Extraire plusieurs contours (level sets) sur le maillage à partir de distances cibles (entrées manuelles)
    pour la représentation "sum" et "diff" séparément.
  - Pour chaque représentation, traiter chaque contour comme une classe, puis appliquer LDA pour réduire
    la dimension des points à 2D et visualiser la séparation entre les différents contours.
  
Auteur : Malek DINARI
"""

import streamlit as st
import numpy as np
import trimesh
import torch
import heapq
from scipy.sparse import lil_matrix
import tempfile
import plotly.graph_objs as go
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import os

st.title("Analyse LDA sur la représentation bipolaire")
st.markdown("""
Cette application vous permet d'extraire la représentation bipolaire d'un maillage 3D (à l'aide des repères des yeux intérieurs)
et d'extraire plusieurs contours (level sets) basés sur des distances cibles pour la somme et la différence.
Chaque contour est considéré comme une classe, et LDA est utilisé pour réduire la dimension à 2D afin de visualiser 
la séparation entre ces classes.
""")

#######################################
# Définir l'appareil CUDA si disponible
#######################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write("Appareil utilisé :", device)

#######################################
# Fonctions de lecture et de visualisation
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

def plot_lda_projection(X_proj, labels, title="Projection LDA"):
    """
    Afficher la projection 2D obtenue par LDA.
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], c=labels, cmap='viridis', s=10)
    ax.set_title(title)
    ax.set_xlabel("LDA 1")
    ax.set_ylabel("LDA 2")
    fig.colorbar(scatter, ax=ax, label="Classe (indice du contour)")
    st.pyplot(fig)

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
# Extraction des contours et LDA sur ces contours
#######################################
def extract_contour(mesh, potentials, target_value, tol=0.5):
    indices = np.where(np.abs(potentials - target_value) < tol)[0]
    st.write(f"[INFO] Extraction du contour: {len(indices)} points trouvés pour cible {target_value} avec tolérance {tol}")
    return mesh.vertices[indices]

def run_lda_on_contours(contours_list, target_values):
    """
    Combine tous les points de contours en un dataset, avec labels indiquant l'indice de la cible.
    Applique LDA pour réduire la dimension à 2 et retourne la projection ainsi que les labels.
    """
    X = []
    y = []
    for i, contour in enumerate(contours_list):
        # Chaque point dans le contour est étiqueté avec l'indice i
        X.append(contour)
        y.append(np.full(contour.shape[0], i))
    X = np.vstack(X)
    y = np.concatenate(y)
    
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_proj = lda.fit_transform(X, y)
    return X_proj, y

#######################################
# Interface Streamlit pour LDA
#######################################
st.sidebar.header("Chargement du fichier")
uploaded_mesh = st.sidebar.file_uploader("Maillage (OBJ)", type=["obj"])
uploaded_bnd = st.sidebar.file_uploader("Repères (.bnd)", type=["bnd"])

st.sidebar.header("Indices des repères (pour les yeux intérieurs)")
ref_left = st.sidebar.number_input("Indice œil intérieur gauche", value=0, step=1)
ref_right = st.sidebar.number_input("Indice œil intérieur droit", value=8, step=1)

st.sidebar.header("Paramètres d'extraction des contours")
contour_tol = st.sidebar.slider("Tolérance pour extraction de contour", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# Entrées de distances cibles pour level sets pour "sum" et "diff"
source_sum_str = st.sidebar.text_input("Target distances (sum)", 
                                         value="97,108,119,130,141,152,163,174,185,196,207,218,229")
source_diff_str = st.sidebar.text_input("Target distances (diff)", 
                                          value="18,19,20,21,22,23,24,25,26,27,28,29,30")

try:
    target_distances_sum = [float(x.strip()) for x in source_sum_str.split(',') if x.strip() != ""]
except Exception as e:
    st.error(f"Erreur de parsing des distances (sum): {e}")
    target_distances_sum = []
try:
    target_distances_diff = [float(x.strip()) for x in source_diff_str.split(',') if x.strip() != ""]
except Exception as e:
    st.error(f"Erreur de parsing des distances (diff): {e}")
    target_distances_diff = []

if st.sidebar.button("Exécuter l'analyse LDA"):
    if uploaded_mesh and uploaded_bnd:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp:
            tmp.write(uploaded_mesh.getvalue())
            mesh_path = tmp.name
        mesh = trimesh.load(mesh_path, process=True)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)


        st.write("[INFO] Lecture des repères...")
        landmarks = landmarks = read_bnd_file(uploaded_bnd.getvalue())
        # Note: here we use the uploaded_bnd directly:
        landmarks = read_bnd_file(uploaded_bnd.getvalue())
        if landmarks is None:
            st.error("Erreur lors de la lecture du fichier .bnd")
        else:
            st.write("[INFO] 5 premières lignes des repères :")
            st.dataframe(landmarks[:5])
            # Utiliser les repères pour les yeux intérieurs
            ref_left_point = landmarks[int(ref_left)]
            ref_right_point = landmarks[int(ref_right)]
            idx_left = find_nearest_vertex(vertices, ref_left_point)
            idx_right = find_nearest_vertex(vertices, ref_right_point)
            st.write(f"[INFO] Indices trouvés : gauche={idx_left}, droite={idx_right}")
            
            # Calculer la représentation bipolaire pour ce maillage
            somme, diff = compute_bipolar_representation(vertices, faces, idx_left, idx_right)
            
            # Extraire les contours (level sets) pour "sum" et "diff"
            contours_sum = []
            for c in target_distances_sum:
                cs = extract_contour(mesh, somme.numpy(), c, tol=contour_tol)
                if cs.size > 0:
                    contours_sum.append(cs)
            contours_diff = []
            for c in target_distances_diff:
                cd = extract_contour(mesh, diff.numpy(), c, tol=contour_tol)
                if cd.size > 0:
                    contours_diff.append(cd)
            
            st.write(f"[INFO] {len(contours_sum)} contours extraits pour 'sum'")
            st.write(f"[INFO] {len(contours_diff)} contours extraits pour 'diff'")
            
            # Exécuter LDA sur les contours pour "sum"
            if len(contours_sum) > 1:
                X_proj_sum, labels_sum = run_lda_on_contours(contours_sum, target_distances_sum)
                st.write("Projection LDA (sum):")
                plot_lda_projection(X_proj_sum, labels_sum, title="Projection LDA sur 'sum'")
            else:
                st.write("Pas assez de contours 'sum' pour LDA.")
            
            # Exécuter LDA sur les contours pour "diff"
            if len(contours_diff) > 1:
                X_proj_diff, labels_diff = run_lda_on_contours(contours_diff, target_distances_diff)
                st.write("Projection LDA (diff):")
                plot_lda_projection(X_proj_diff, labels_diff, title="Projection LDA sur 'diff'")
            else:
                st.write("Pas assez de contours 'diff' pour LDA.")
    else:
        st.error("Veuillez uploader un maillage (.obj) et un fichier de repères (.bnd).")
