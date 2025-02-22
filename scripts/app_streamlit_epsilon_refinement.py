#!/usr/bin/env python
"""
Application Streamlit : Raffinement adaptatif d'epsilon pour contours géodésiques (P1 + P2)

Modifications :
1. Les cibles (C constantes) pour l'extraction des contours équipotentiels sont désormais : 65, 85, 105, 125, 145.
2. La règle adaptative d'ε est rétablie à la version d'origine, définie par :
   ε(t) = base_eps * (1 + (t/np.max(TARGETS))**(-1.5))
3. Ajout d'une visualisation du plan XY (slice en Z) avec tolérance ± (z_tol par défaut 0.10) et
   d'un graphique de la fonction initiale ε = f(P).
   
La formulation mathématique (en français) de la règle adaptative d'ε est expliquée ci-dessus.
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

# ---------------------- Configuration -------------------------
# Mise à jour des cibles pour les contours équipotentiels
TARGETS = [65, 85, 105, 125, 145]
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

# ---------------------- Interface améliorée ---------------------------
st.title("🌐 Extraction de contours géodésiques adaptatifs")
st.markdown("""
**Fonctionnalités clés** :
- Extraction initiale avec ε fixe (0.25)
- Vue en plan XY (slice en Z) avec tolérance ± (z_tol par défaut 0.10)
- Graphique de la fonction initiale ε = f(P)
- Raffinement adaptatif d'ε (règle d'origine)
""")

# ---------------------- Chargement des fichiers ---------------------------
with st.sidebar:
    st.header("📁 Chargement des fichiers")
    bnd_file = st.file_uploader("Fichier .bnd", type=["bnd"])
    obj_file = st.file_uploader("Fichier .obj", type=["obj"])

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
        # Chargement des landmarks depuis le fichier .bnd (format attendu : colonnes, la 1ère colonne est ignorée)
        landmarks = np.loadtxt(bnd_file, delimiter=None)
        eye_left = landmarks[0][1:4]
        eye_right = landmarks[8][1:4]

    with st.spinner("Calcul des distances géodésiques..."):
        adj = build_sparse_adjacency(mesh.vertices, mesh.faces)
        idx_left = mesh.kdtree.query(eye_left)[1]
        idx_right = mesh.kdtree.query(eye_right)[1]
        d1 = dijkstra(adj, idx_left)
        d2 = dijkstra(adj, idx_right)
        P = (d1 + d2).cpu().numpy()

    # ---------------------- Paramètres d'extraction ---------------------------
    with st.sidebar:
        st.header("⚙️ Paramètres d'extraction")
        # Pour la vue en plan XY, on définit la position du plan Z et son épaisseur
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

    # Affichage des métriques de base
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temps chargement", f"{st.session_state.load_time:.2f}s")
    with col2:
        st.metric("Temps adjacence", f"{st.session_state.adj_time:.2f}s")
    with col3:
        st.metric("Temps Dijkstra", f"{st.session_state.dijkstra_time:.2f}s")
    
    # Visualisation 3D du maillage et des contours extraits initialement
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
    # Règle d'origine : 
    # ε(t) = base_eps * (1 + (t/np.max(TARGETS))**(-1.5))
    base_eps = 0.1
    epsilons = [base_eps * (1 + (t/np.max(TARGETS))**(-1.5)) for t in TARGETS]
    
    # Extraction adaptative avec ces valeurs de ε
    adaptive_contours = []
    start_time = time.time()
    for target, eps in zip(TARGETS, epsilons):
        mask = np.abs(P - target) < eps
        adaptive_contours.append((target, mask.sum(), mesh.vertices[mask]))
    metrics['adaptive_time'] = time.time() - start_time

    # Visualisation 3D des contours adaptatifs
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
    
    # Graphique de la fonction de raffinement adaptatif ε = f(P) avec la règle d'origine
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
    
    # Tableau comparatif des extractions
    st.header("Comparaison des extractions")
    comparison_data = {
        'Cible': TARGETS,
        'Pts (fixe)': [c[1] for c in initial_contours],
        'Pts (adaptatif)': [c[1] for c in adaptive_contours],
        'ε adaptatif': epsilons
    }
    st.dataframe(comparison_data, use_container_width=True)
    
    # Affichage des métriques finales
    st.subheader("⏱ Métriques de performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Temps extraction fixe", f"{metrics['extraction_time']:.2f}s")
    with col2:
        st.metric("Temps extraction adaptative", f"{metrics['adaptive_time']:.2f}s")
    
else:
    st.error("Veuillez uploader les fichiers .bnd et .obj.")