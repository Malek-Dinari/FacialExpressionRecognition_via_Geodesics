import streamlit as st
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
import tempfile

from mesh_utils import load_mesh, icp_registration, convert_o3d_to_plotly
from landmarks import read_bnd

st.title("Alignement de Maillages 3D avec ICP")
st.markdown("""
Ce module permet d'aligner deux maillages 3D (par ex. deux visages du dataset BU-3DFE) en utilisant l'algorithme ICP.
Vous pouvez uploader le maillage source et le maillage cible, ajuster les paramètres, et visualiser le résultat.
""")

# Barre latérale pour les paramètres ICP
st.sidebar.header("Paramètres ICP")
threshold = st.sidebar.slider("Distance seuil", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
max_iter = st.sidebar.slider("Nombre max d'itérations", min_value=10, max_value=200, value=50, step=10)

# Chargement des fichiers
st.sidebar.header("Charger les fichiers")
uploaded_source = st.sidebar.file_uploader("Maillage source (OBJ, PLY, STL, etc.)", type=["obj", "ply", "stl"])
uploaded_target = st.sidebar.file_uploader("Maillage cible (OBJ, PLY, STL, etc.)", type=["obj", "ply", "stl"])
uploaded_landmarks = st.sidebar.file_uploader("Fichier de landmarks (.bnd)", type=["bnd"])

if st.sidebar.button("Exécuter l'alignement ICP"):
    if uploaded_source is not None and uploaded_target is not None:
        # Sauvegarder les fichiers uploadés dans un répertoire temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as source_temp:
            source_temp.write(uploaded_source.getvalue())
            source_path = source_temp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as target_temp:
            target_temp.write(uploaded_target.getvalue())
            target_path = target_temp.name

        st.write("Chargement des maillages...")
        source_mesh = load_mesh(source_path)
        target_mesh = load_mesh(target_path)

        st.write("Exécution de l'alignement ICP...")
        transformation, eval_result = icp_registration(source_mesh, target_mesh, threshold=threshold, max_iteration=max_iter)

        st.write("Matrice de transformation obtenue :")
        st.write(transformation)

        # Appliquer la transformation à la source
        source_mesh.transform(transformation)

        # Convertir les maillages pour Plotly
        src_vertices, src_triangles = convert_o3d_to_plotly(source_mesh)
        tgt_vertices, tgt_triangles = convert_o3d_to_plotly(target_mesh)

        # Création de la figure Plotly
        fig = go.Figure()

        # Afficher le maillage cible (en bleu, semi-transparent)
        fig.add_trace(go.Mesh3d(
            x=tgt_vertices[:, 0],
            y=tgt_vertices[:, 1],
            z=tgt_vertices[:, 2],
            i=tgt_triangles[:, 0],
            j=tgt_triangles[:, 1],
            k=tgt_triangles[:, 2],
            opacity=0.5,
            color='blue',
            name='Maillage cible'
        ))
        # Afficher le maillage source aligné (en rouge)
        fig.add_trace(go.Mesh3d(
            x=src_vertices[:, 0],
            y=src_vertices[:, 1],
            z=src_vertices[:, 2],
            i=src_triangles[:, 0],
            j=src_triangles[:, 1],
            k=src_triangles[:, 2],
            opacity=0.9,
            color='red',
            name='Maillage source aligné'
        ))
        fig.update_layout(
            title="Résultat de l'alignement ICP",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Si un fichier de landmarks est fourni, le lire et l'afficher sur le maillage source
        if uploaded_landmarks is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bnd") as land_temp:
                land_temp.write(uploaded_landmarks.getvalue())
                land_path = land_temp.name
            landmarks = read_bnd(land_path)
            if landmarks is not None:
                st.write("Landmarks chargés :")
                st.write(landmarks)
                # Visualisation des landmarks sur le maillage source
                fig_land = go.Figure()
                fig_land.add_trace(go.Mesh3d(
                    x=src_vertices[:, 0],
                    y=src_vertices[:, 1],
                    z=src_vertices[:, 2],
                    i=src_triangles[:, 0],
                    j=src_triangles[:, 1],
                    k=src_triangles[:, 2],
                    opacity=0.5,
                    color='gray',
                    name='Maillage source'
                ))
                fig_land.add_trace(go.Scatter3d(
                    x=landmarks[:, 0],
                    y=landmarks[:, 1],
                    z=landmarks[:, 2],
                    mode='markers',
                    marker=dict(size=5, color='red'),
                    name='Landmarks'
                ))
                fig_land.update_layout(
                    title="Landmarks sur le maillage source",
                    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
                )
                st.plotly_chart(fig_land, use_container_width=True)
    else:
        st.error("Veuillez uploader les deux maillages source et cible.")
