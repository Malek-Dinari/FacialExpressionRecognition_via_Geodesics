#!/usr/bin/env python
"""
app_streamlit_acp_bipolar_bspline.py

Cette application Streamlit permet de :
  - Charger un maillage 3D (face) et son fichier de repères (.bnd) du dataset BU3DFE.
  - Extraire les repères correspondant aux bords intérieurs des yeux (indices 0 et 8 par défaut).
  - Calculer la représentation bipolaire (potentiels géodésiques "sum" et "diff") à l'aide de ces repères.
  - Extraire plusieurs contours (level sets) sur le maillage via des distances cibles (entrées manuelles) et une tolérance.
  - Pour chaque contour, réordonner les points par longueur d'arc, fermer le contour, ajuster une B‑Spline et échantillonner un nombre fixe de points,
    constituant ainsi une représentation continue (vecteur de caractéristiques).
  - Appliquer PCA sur l'ensemble des vecteurs de caractéristiques pour réduire la dimension à 2D et visualiser la séparation des contours.
  - Visualiser séparément les contours "sum", les contours "diff", et une superposition des deux.
  
L'idée ici est de comparer la séparation obtenue pour les représentations "sum" et "diff".

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
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev

import os

st.title("Analyse avec ACP/LDA sur la représentation bipolaire")
st.markdown("""
Cette application vous permet d'extraire la représentation bipolaire d'un maillage 3D (à l'aide des repères des yeux intérieurs)
et d'extraire plusieurs contours (level sets) basés sur des distances cibles pour la *somme* et la *différence*.
Chaque contour est considéré comme une classe, et PCA/LDA sont utilisé pour réduire la dimension à 2D afin de visualiser les importantes features (directions qui définissent le meilleur plan de projection)
la séparation entre ces classes (discrimination).
""")


#######################################
# Définir l'appareil CUDA si disponible
#######################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write("Appareil utilisé :", device)


#######################################
# Fonctions de visualisation PCA/LDA
#######################################
def plot_lda_projection(X_proj, labels, title="Projection LDA"):
    """
    Afficher la projection 2D obtenue par PCA (ou LDA).
    X_proj est un tableau de dimension (N, 2) et labels est un vecteur d'étiquettes.
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], c=labels, cmap='viridis', s=10)
    ax.set_title(title)
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    fig.colorbar(scatter, ax=ax, label="Classe (indice du contour)")
    st.pyplot(fig)

def plot_contours(contours_list, target_list, title="Contours"):
    """
    Afficher plusieurs ensembles de points (nuages) représentant les contours.
    Chaque contour est tracé avec une couleur différente et étiqueté par sa distance cible.
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


def plot_interpolated_curves(contours_interp, target_list, title="Courbes interpolées"):
    fig = go.Figure()
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'magenta', 'brown', 'cyan']
    for i, curve in enumerate(contours_interp):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=curve[:, 0],
            y=curve[:, 1],
            z=curve[:, 2],
            mode='lines+markers',
            marker=dict(size=3, color=color),
            line=dict(color=color),
            name=f'Courbe C={target_list[i]:.2f}'
        ))
    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    st.plotly_chart(fig, use_container_width=True)


def plot_combined_contours(contours_sum, contours_diff, title="Contours combinés"):
    """
    Superposer les contours "sum" et "diff" dans un seul graphique.
    """
    fig = go.Figure()
    colors_sum = ['red', 'orange', 'green', 'blue']
    colors_diff = ['purple', 'magenta', 'brown', 'cyan']
    for i, contour in enumerate(contours_sum):
        fig.add_trace(go.Scatter3d(
            x=contour[:, 0],
            y=contour[:, 1],
            z=contour[:, 2],
            mode='markers',
            marker=dict(size=3, color=colors_sum[i % len(colors_sum)]),
            name=f'Sum C={i}'
        ))
    for i, contour in enumerate(contours_diff):
        fig.add_trace(go.Scatter3d(
            x=contour[:, 0],
            y=contour[:, 1],
            z=contour[:, 2],
            mode='markers',
            marker=dict(size=3, color=colors_diff[i % len(colors_diff)]),
            name=f'Diff C={i}'
        ))
    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    return fig

#######################################
# Fonctions de lecture du fichier BND et visualisation
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
            name=f'intérieur de l\'Œil gauche (index {left_index})'
        ))
    if right_index is not None and right_index < landmarks.shape[0]:
        fig.add_trace(go.Scatter3d(
            x=[landmarks[right_index, 0]],
            y=[landmarks[right_index, 1]],
            z=[landmarks[right_index, 2]],
            mode='markers',
            marker=dict(size=10, color='orange'),
            name=f'intérieur de l\'Œil droit (index {right_index})'
        ))
    fig.update_layout(title="Maillage et repères", scene=dict(aspectmode='data'))
    return fig

#######################################
# Fonctions pour réordonner et interpoler les contours
#######################################
def order_contour_points(points, contour_type="sum"):
    """
    Order points based on contour type.
    - For sum level sets, use endpoint-based ordering.
    - For diff level sets, split into left and right parts and order separately.
    """
    if len(points) <= 1:
        return points.copy()
    
    if contour_type == "sum":
        # Use endpoint-based ordering for sum level sets
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
    
    elif contour_type == "diff":
        # For diff level sets, split into left and right parts
        centroid = np.mean(points, axis=0)
        left_points = points[points[:, 0] < centroid[0]]
        right_points = points[points[:, 0] >= centroid[0]]
        
        # Order left and right parts separately
        left_ordered = left_points[np.argsort(left_points[:, 1])]  # Sort by y-coordinate
        right_ordered = right_points[np.argsort(right_points[:, 1])]  # Sort by y-coordinate
        
        # Combine left and right parts
        return np.vstack((left_ordered, right_ordered[::-1]))
    
    else:
        raise ValueError("Invalid contour type. Use 'sum' or 'diff'.")

def fit_bspline_curve(points, n_samples=50, s=0, contour_type="sum"):
    """
    Fit B-spline with adaptive handling for sum and diff level sets.
    """
    if len(points) < 4:
        return points
    
    points_ordered = order_contour_points(points, contour_type=contour_type)
    
    # Check if contour is closed (only for sum level sets)
    is_closed = contour_type == "sum" and np.linalg.norm(points_ordered[0] - points_ordered[-1]) < 1e-3
    k = 3 if len(points_ordered) >= 4 else min(3, len(points_ordered)-1)
    
    try:
        # Arc length parametrization
        dists = np.linalg.norm(np.diff(points_ordered, axis=0), axis=1)
        arc_length = np.concatenate(([0], np.cumsum(dists)))
        arc_length_norm = arc_length / arc_length[-1] if arc_length[-1] != 0 else np.linspace(0, 1, len(points_ordered))
        
        # Fit with periodic if closed (only for sum level sets)
        tck, u = splprep(points_ordered.T, u=arc_length_norm, s=s, k=k, per=is_closed)
        u_new = np.linspace(0, 1, n_samples)
        if is_closed:
            u_new = u_new[:-1]  # Avoid duplicate point
            
        curve = np.array(splev(u_new, tck)).T
        return curve
    except Exception as e:
        st.error(f"B-spline error: {str(e)}")
        return points_ordered

#######################################
# Fonction PCA sur les vecteurs de caractéristiques
#######################################
def run_pca_on_curves(curve_features):
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(curve_features)
    return X_proj, pca

#######################################
# Extraction des contours et création de vecteurs B‑Spline
#######################################
def process_contours(mesh, potentials, target_distances, tol=0.5, n_samples=50, spline_s=0):
    """
    Pour chaque distance cible, extraire le contour, ajuster une B‑Spline et aplatir le résultat pour obtenir un vecteur.
    Retourne:
      - features: tableau de dimension (N, 3*n_samples)
      - labels: les distances cibles
      - contours_interp: liste des courbes interpolées pour visualisation.
    """
    features = []
    labels = []
    contours_interp = []
    for c in target_distances:
        pts = extract_contour(mesh, potentials, c, tol=tol)
        if pts.shape[0] == 0:
            continue
        curve = fit_bspline_curve(pts, n_samples=n_samples, s=spline_s)
        contours_interp.append(curve)
        feat = curve.flatten()
        features.append(feat)
        labels.append(c)
    if features:
        features = np.vstack(features)
    else:
        features = np.array([])
    return features, np.array(labels), contours_interp

#######################################
# Fonctions pour la représentation bipolaire
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
# Extraction d'un contour
#######################################
def extract_contour(mesh, potentials, target_value, tol=0.5):
    indices = np.where(np.abs(potentials - target_value) < tol)[0]
    st.write(f"[INFO] Extraction du contour: {len(indices)} points trouvés pour cible {target_value} avec tolérance {tol}")
    return mesh.vertices[indices]

#######################################
# Interface Streamlit
#######################################
st.sidebar.header("Chargement des fichiers")
uploaded_mesh = st.sidebar.file_uploader("Maillage (OBJ)", type=["obj"])
uploaded_bnd = st.sidebar.file_uploader("Repères (.bnd)", type=["bnd"])

st.sidebar.header("Extraction des 2 Landmarks (points de référence) intérieurs des yeux)")
ref_left = st.sidebar.number_input("Indice œil intérieur gauche", value=0, step=1)
ref_right = st.sidebar.number_input("Indice œil intérieur droit", value=8, step=1)

st.sidebar.header("Paramètres d'extraction des contours")
contour_tol = st.sidebar.slider("Tolérance pour extraction de contour", min_value=0.1, max_value=2.0, value=0.5, step=0.01)
n_samples = st.sidebar.number_input("Nombre de points pour B-Spline", value=50, step=1)
spline_s = st.sidebar.number_input("Paramètre de lissage B-Spline", value=0.0, step=0.1)

st.sidebar.header("Distances cibles pour les level sets")
sum_str = st.sidebar.text_input("Target distances (sum)", 
                                  value="50,66,98,114,132,164,181")
diff_str = st.sidebar.text_input("Target distances (diff)", 
                                   value="7,20,25,28,30,32,41,42,43")
try:
    target_distances_sum = [float(x.strip()) for x in sum_str.split(',') if x.strip() != ""]
except Exception as e:
    st.error(f"Erreur de parsing des distances (sum): {e}")
    target_distances_sum = []
try:
    target_distances_diff = [float(x.strip()) for x in diff_str.split(',') if x.strip() != ""]
except Exception as e:
    st.error(f"Erreur de parsing des distances (diff): {e}")
    target_distances_diff = []

if st.sidebar.button("Exécuter l'analyse PCA sur les contours"):
    if uploaded_mesh and uploaded_bnd:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp:
            tmp.write(uploaded_mesh.getvalue())
            mesh_path = tmp.name
        mesh = trimesh.load(mesh_path, process=True)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        landmarks = read_bnd_file(uploaded_bnd.getvalue())
        if landmarks is None:
            st.error("Erreur lors de la lecture du fichier .bnd")
        else:
            st.write("[INFO] 5 premières lignes des repères :")
            st.dataframe(landmarks[:5])
            ref_left_point = landmarks[int(ref_left)]
            ref_right_point = landmarks[int(ref_right)]
            idx_left = find_nearest_vertex(vertices, ref_left_point)
            idx_right = find_nearest_vertex(vertices, ref_right_point)
            st.write(f"[INFO] Indices trouvés : gauche={idx_left}, droite={idx_right}")
            
            # Calcul de la représentation bipolaire
            somme, diff = compute_bipolar_representation(vertices, faces, idx_left, idx_right)
            
            # Extraction des contours pour "sum" et "diff"
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
            
            # Visualisation des contours extraits
            st.write("[INFO] Visualisation des contours 'sum'")
            fig_sum = plot_contours(contours_sum, target_distances_sum, title="Contours (sum)")
            st.plotly_chart(fig_sum, use_container_width=True)
            st.write("[INFO] Visualisation des contours 'diff'")
            fig_diff = plot_contours(contours_diff, target_distances_diff, title="Contours (diff)")
            st.plotly_chart(fig_diff, use_container_width=True)
            
            # Visualisation des courbes interpolées
            st.write("[INFO] Visualisation des courbes interpolées (B-Spline) pour 'sum'")
            interp_curves_sum = [
    fit_bspline_curve(order_contour_points(c, contour_type="sum"), n_samples=n_samples, s=spline_s, contour_type="sum")
    for c in contours_sum
]
            plot_interpolated_curves(interp_curves_sum, target_distances_sum, title="Courbes interpolées (sum)")
            st.write("[INFO] Visualisation des courbes interpolées (B-Spline) pour 'diff'")
            interp_curves_diff = [
    fit_bspline_curve(order_contour_points(c, contour_type="diff"), n_samples=n_samples, s=spline_s, contour_type="diff")
    for c in contours_diff
]
            plot_interpolated_curves(interp_curves_diff, target_distances_diff, title="Courbes interpolées (diff)")
            
            # Superposition des contours "sum" et "diff"
            st.write("[INFO] Visualisation combinée des contours 'sum' et 'diff'")
            fig_combined = plot_combined_contours(contours_sum, contours_diff, title="Contours combinés")
            st.plotly_chart(fig_combined, use_container_width=True)
            
            # Création des vecteurs de caractéristiques par B-Spline pour chaque contour
            def process_curve_features(contours, n_samples, spline_s, target_list):
                features = []
                labels = []
                for i, curve_points in enumerate(contours):
                    ordered = order_contour_points(curve_points)
                    smooth_curve = fit_bspline_curve(ordered, n_samples=n_samples, s=spline_s)
                    features.append(smooth_curve.flatten())
                    labels.append(target_list[i])
                if features:
                    return np.vstack(features), np.array(labels)
                else:
                    return np.array([]), np.array([])
            
            features_sum, labels_sum, _ = process_contours(mesh, somme.numpy(), target_distances_sum, tol=contour_tol, n_samples=n_samples, spline_s=spline_s)
            features_diff, labels_diff, _ = process_contours(mesh, diff.numpy(), target_distances_diff, tol=contour_tol, n_samples=n_samples, spline_s=spline_s)
            
            st.write(f"[INFO] {features_sum.shape[0]} vecteurs de caractéristiques obtenus pour 'sum'")
            st.write(f"[INFO] {features_diff.shape[0]} vecteurs de caractéristiques obtenus pour 'diff'")
            
            if features_sum.size > 0:
                pca_sum = PCA(n_components=2)
                X_proj_sum = pca_sum.fit_transform(features_sum)
                st.write("Projection PCA sur les contours 'sum':")
                plot_lda_projection(X_proj_sum, labels_sum, title="Projection PCA (sum)")
            else:
                st.write("Pas assez de contours 'sum' pour PCA.")
            
            if features_diff.size > 0:
                pca_diff = PCA(n_components=2)
                X_proj_diff = pca_diff.fit_transform(features_diff)
                st.write("Projection PCA sur les contours 'diff':")
                plot_lda_projection(X_proj_diff, labels_diff, title="Projection PCA (diff)")
            else:
                st.write("Pas assez de contours 'diff' pour PCA.")
    else:
        st.error("Veuillez uploader un maillage (.obj) et un fichier de repères (.bnd).")
