#!/usr/bin/env python
"""
Application Streamlit : Raffinement adaptatif d'epsilon pour contours géodésiques (P1 + P2)
avec option de représentation tripolaire et détection automatique de la pointe du nez

Modifications :
1. Les cibles (TARGETS) pour l'extraction des contours équipotentiels sont personnalisables via une entrée texte (défaut : "105, 135, 165, 195, 225").
2. Option de représentation : Bipolaire (P = d1 + d2) ou Tripolaire (P = coef_d1*d1 + coef_d2*d2 + coef_d3*d3).
3. Pour la représentation tripolaire, la pointe du nez est automatiquement détectée comme le vertex ayant la plus haute courbure dans la région centrale du visage.
4. Les visualisations du maillage 3D, du slice XY et du graphique ε = f(P) sont conservées.

La règle adaptative d'ε utilisée est :
    ε(t) = base_eps * (1 + (t/np.max(TARGETS))**(-1.5))
avec base_eps = 0.1.
Auteur : Malek DINARI
"""

import streamlit as st
import os
import numpy as np
import trimesh
import torch
import plotly.graph_objs as go
import heapq
import time
from scipy.sparse import lil_matrix
import tempfile

# ---------------------- Paramètres globaux -------------------------
INITIAL_EPSILON = 0.25  # Tolérance fixe initiale pour extraction
COLORS = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']

# ---------------------- Fonctions de base -------------------------

def load_obj_mesh(file_obj):
    """Charge un fichier OBJ et renvoie le maillage (avec chronométrage)."""
    start_time = time.time()
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp_file:
        tmp_file.write(file_obj.read())
        tmp_file_path = tmp_file.name
    mesh = trimesh.load(tmp_file_path, process=True)
    os.unlink(tmp_file_path)
    st.session_state.load_time = time.time() - start_time
    return mesh

def build_sparse_adjacency(vertices, faces):
    """Construit une matrice d'adjacence pondérée."""
    start_time = time.time()
    adj = lil_matrix((len(vertices), len(vertices)), dtype=np.float32)
    for face in faces:
        for i, j in [(0,1), (1,2), (0,2)]:
            vi, vj = face[i], face[j]
            dist = np.linalg.norm(vertices[vi] - vertices[vj])
            adj[vi, vj] = dist
            adj[vj, vi] = dist
    st.session_state.adj_time = time.time() - start_time
    return adj.tocsr()

def dijkstra(adj, start):
    """Calcul des distances géodésiques avec l'algorithme de Dijkstra."""
    start_time = time.time()
    distances = np.full(adj.shape[0], np.inf)
    distances[start] = 0
    heap = [(0.0, start)]
    while heap:
        dist, u = heapq.heappop(heap)
        if dist > distances[u]:
            continue
        for v, weight in zip(adj[u].indices, adj[u].data):
            new_dist = dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    st.session_state.dijkstra_time = time.time() - start_time
    return torch.tensor(distances, dtype=torch.float32)

def find_nose_tip(mesh):
    """
    Recherche automatique de la pointe du nez.
    On calcule une mesure de courbure pour chaque vertex (en utilisant la variation des normales avec ses voisins)
    et on restreint la recherche aux vertices proches du centre du maillage.
    Le vertex avec la plus haute courbure parmi ces candidats est considéré comme la pointe du nez.
    """
    vertices = mesh.vertices
    normals = mesh.vertex_normals
    center = np.mean(vertices, axis=0)
    # Construire la liste des voisins pour chaque vertex
    neighbors = {i: set() for i in range(len(vertices))}
    for face in mesh.faces:
        for i in range(3):
            neighbors[face[i]].update([face[(i+1)%3], face[(i+2)%3]])
    curvature = np.zeros(len(vertices))
    for i in range(len(vertices)):
        neigh_indices = list(neighbors[i])
        if len(neigh_indices) > 0:
            dots = np.dot(normals[i], normals[neigh_indices].T)
            curvature[i] = np.mean(1 - dots)
        else:
            curvature[i] = 0
    dists_to_center = np.linalg.norm(vertices - center, axis=1)
    threshold = 0.5 * np.max(dists_to_center)
    candidate_indices = np.where(dists_to_center < threshold)[0]
    if len(candidate_indices) == 0:
        return np.argmax(curvature)
    candidate_curvatures = curvature[candidate_indices]
    nose_tip_index = candidate_indices[np.argmax(candidate_curvatures)]
    return nose_tip_index

# ---------------------- Interface Streamlit ---------------------------
st.title("🌐 Extraction de contours géodésiques adaptatifs")
st.markdown("""
**Fonctionnalités clés** :
- Extraction initiale avec ε fixe (0.25)
- Vue en plan XY (slice en Z) avec tolérance ± (z_tol par défaut 0.10)
- Graphique de la fonction initiale ε = f(P)
- Choix entre représentation Bipolaire ou Tripolaire  
  (pour Tripolaire, la pointe du nez est détectée automatiquement comme le point de plus haute courbure dans la zone centrale)
""")

# ---------------------- Chargement des fichiers ---------------------------
with st.sidebar:
    st.header("📁 Chargement des fichiers")
    bnd_file = st.file_uploader("Fichier .bnd", type=["bnd"])
    obj_file = st.file_uploader("Fichier .obj", type=["obj"])

# ---------------------- Paramètres de représentation ---------------------------
with st.sidebar:
    st.header("🔧 Paramètres de représentation")
    representation_type = st.radio("Type de représentation", ["Bipolaire", "Tripolaire"])
    coef_d1 = st.number_input("Coefficient pour d1", value=1.0, step=0.1)
    coef_d2 = st.number_input("Coefficient pour d2", value=1.0, step=0.1)
    if representation_type == "Tripolaire":
        auto_nose = st.checkbox("Détecter automatiquement la pointe du nez", value=True)
        if not auto_nose:
            tripolar_index = st.number_input("Indice repère additionnel (nez)", value=16, step=1)
        coef_d3 = st.number_input("Coefficient pour d3", value=1.0, step=0.1)
    # Entrée pour les cibles TARGETS, modifiables par l'utilisateur
    targets_input = st.text_input("Cibles pour extraction (séparées par des virgules)", value="105, 135, 165, 195, 225")
    try:
        TARGETS = [float(x.strip()) for x in targets_input.split(",") if x.strip() != ""]
    except:
        TARGETS = [105, 135, 165, 195, 225]

if bnd_file and obj_file:
    metrics = {
        'load_time': 0,
        'adj_time': 0,
        'dijkstra_time': 0,
        'extraction_time': 0,
        'adaptive_time': 0
    }

    with st.spinner("Chargement du maillage..."):
        mesh = load_obj_mesh(obj_file)
        # Chargement des landmarks depuis le fichier .bnd (les 3 colonnes suivantes de la 1ère colonne ignorée)
        landmarks = np.loadtxt(bnd_file, delimiter=None)
        eye_left = landmarks[0][1:4]
        eye_right = landmarks[8][1:4]

    with st.spinner("Calcul des distances géodésiques..."):
        adj = build_sparse_adjacency(mesh.vertices, mesh.faces)
        idx_left = mesh.kdtree.query(eye_left)[1]
        idx_right = mesh.kdtree.query(eye_right)[1]
        d1 = dijkstra(adj, idx_left)
        d2 = dijkstra(adj, idx_right)
        if representation_type == "Bipolaire":
            P = (coef_d1 * d1 + coef_d2 * d2).cpu().numpy()
        else:
            if auto_nose:
                nose_index = find_nose_tip(mesh)
                st.write("Pointe du nez détectée à l'index :", nose_index)
            else:
                nose_index = int(tripolar_index)
            d3 = dijkstra(adj, nose_index)
            P = (coef_d1 * d1 + coef_d2 * d2 + coef_d3 * d3).cpu().numpy()

    # ---------------------- Paramètres d'extraction ---------------------------
    with st.sidebar:
        st.header("⚙️ Paramètres d'extraction")
        z_slice = st.slider("Position du plan Z", 
                            float(mesh.bounds[0][2]), 
                            float(mesh.bounds[1][2]), 
                            float(np.median(mesh.vertices[:,2])))
        z_tol = st.slider("Tolérance sur Z pour le slice", 0.01, 1.0, 0.10)
    
    # ---------------------- Extraction initiale avec ε fixe ---------------------------
    st.header("1. Extraction initiale (ε fixe)")
    initial_contours = []
    start_time = time.time()
    for target in TARGETS:
        mask = np.abs(P - target) < INITIAL_EPSILON
        initial_contours.append((target, mask.sum(), mesh.vertices[mask]))
    metrics['extraction_time'] = time.time() - start_time

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temps chargement", f"{st.session_state.load_time:.2f}s")
    with col2:
        st.metric("Temps adjacence", f"{st.session_state.adj_time:.2f}s")
    with col3:
        st.metric("Temps Dijkstra", f"{st.session_state.dijkstra_time:.2f}s")
    
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Mesh3d(
        x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
        opacity=0.1, color='lightgray', name='Maillage'
    ))
    for i, (target, count, pts) in enumerate(initial_contours):
        fig_3d.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=3, color=COLORS[i]),
            name=f'Cible {target} ({count} pts)'
        ))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # ---------------------- Vue en plan XY (slice en Z) ---------------------------
    st.header("2. Vue en plan XY de la tranche (z-slice)")
    slice_mask = np.abs(mesh.vertices[:,2] - z_slice) < z_tol
    vertices_slice = mesh.vertices[slice_mask]
    pot_slice = P[slice_mask]
    fig_xy = go.Figure()
    fig_xy.add_trace(go.Scatter(
        x=vertices_slice[:,0], y=vertices_slice[:,1],
        mode='markers', marker=dict(size=4, color=pot_slice, colorscale='Viridis', colorbar=dict(title="Potentiel")),
        name='Points slice XY'
    ))
    fig_xy.update_layout(title=f"Plan XY (z = {z_slice} ± {z_tol})", xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig_xy, use_container_width=True)
    
    # ---------------------- Graphique de la fonction initiale ε = f(P) ---------------------------
    st.header("3. Fonction initiale ε = f(P) (extraction fixe)")
    initial_eps_values = []
    for target in TARGETS:
        mask = np.abs(P - target) < INITIAL_EPSILON
        if mask.sum() > 0:
            mean_err = np.mean(np.abs(P[mask] - target))
        else:
            mean_err = np.nan
        initial_eps_values.append(mean_err)
    fig_initial_eps = go.Figure()
    fig_initial_eps.add_trace(go.Scatter(
        x=TARGETS, y=initial_eps_values,
        mode='lines+markers',
        line=dict(shape='spline', color='orange')
    ))
    fig_initial_eps.update_layout(
        title="Fonction initiale ε = f(P) (extraction fixe)",
        xaxis_title="Potentiel cible",
        yaxis_title="ε initial moyen",
        template='plotly_white'
    )
    st.plotly_chart(fig_initial_eps, use_container_width=True)
    
    # ---------------------- Raffinement adaptatif d'ε selon la règle d'origine ---------------------------
    st.header("4. Raffinement adaptatif d'ε")
    base_eps = 0.1
    epsilons = [base_eps * (1 + (t/np.max(TARGETS))**(-1.5)) for t in TARGETS]
    
    adaptive_contours = []
    start_time = time.time()
    for target, eps in zip(TARGETS, epsilons):
        mask = np.abs(P - target) < eps
        adaptive_contours.append((target, mask.sum(), mesh.vertices[mask]))
    metrics['adaptive_time'] = time.time() - start_time

    fig_adapt = go.Figure()
    fig_adapt.add_trace(go.Mesh3d(
        x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
        opacity=0.1, color='lightgray', name='Maillage'
    ))
    for i, (target, count, pts) in enumerate(adaptive_contours):
        fig_adapt.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=3, color=COLORS[i]),
            name=f'Cible {target} ({count} pts)'
        ))
    st.plotly_chart(fig_adapt, use_container_width=True)
    
    fig_adapt_eps = go.Figure()
    fig_adapt_eps.add_trace(go.Scatter(
        x=TARGETS, y=epsilons,
        mode='lines+markers',
        line=dict(shape='spline', color='purple')
    ))
    fig_adapt_eps.update_layout(
        title="Fonction de raffinement adaptatif ε = f(P)",
        xaxis_title="Potentiel cible",
        yaxis_title="ε adaptatif",
        template='plotly_white'
    )
    st.plotly_chart(fig_adapt_eps, use_container_width=True)
    
    st.header("Comparaison des extractions")
    comparison_data = {
        'Cible': TARGETS,
        'Pts (fixe)': [c[1] for c in initial_contours],
        'Pts (adaptatif)': [c[1] for c in adaptive_contours],
        'ε adaptatif': epsilons
    }
    st.dataframe(comparison_data, use_container_width=True)
    
    st.subheader("⏱ Métriques de performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Temps extraction fixe", f"{metrics['extraction_time']:.2f}s")
    with col2:
        st.metric("Temps extraction adaptative", f"{metrics['adaptive_time']:.2f}s")
    
else:
    st.error("Veuillez uploader les fichiers .bnd et .obj.")
