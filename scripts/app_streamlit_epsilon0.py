#!/usr/bin/env python
"""
Streamlit app: Raffinement d'epsilon in a Three-Pole Geodesic Representation

This app uses:
  - A .bnd file (BU3D-FE landmarks) to obtain the inner eyes (indexes 0 and 8)
  - An .obj mesh file (face mesh)
  - A computed nose top point (via a simple curvature measure)

It then computes geodesic distances from these three reference points,
forms two potentials (sum and alternative difference), and for a fixed z–slice
(i.e. the XY plane), it computes a smooth ε–variation function (from a dense sampling
of 120 potential values) as well as a coarse set of equipotential contours for visualization.

Additionally, it visualizes the sampling of potential values in the xy plane so you can
confirm the ruler (axis) of potential values.

Quantiles/Percentiles:
  - A percentile (e.g. 10th percentile) is used here to compute the tolerance ε at each target.
    For a given target potential t, we calculate |P(x,y) - t| for all (x,y) in the slice,
    then set ε = the chosen percentile of these values.
  - Quantiles (like 10%, 50%, 90%) can be used to choose representative target levels.
  
Author: Malek DINARI
"""

import streamlit as st
import numpy as np
import trimesh
import torch
import plotly.graph_objs as go
import os, time, heapq
from tqdm import tqdm
from scipy.sparse import lil_matrix
from scipy.interpolate import interp1d
import tempfile
from io import StringIO

# ---------------------- Mesh and Geodesic Functions -------------------------

def load_obj_mesh(filepath):
    """Load an OBJ mesh file and extract vertices, faces, and the trimesh object."""
    mesh = trimesh.load(filepath, process=True)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    return vertices, faces, mesh

def build_sparse_adjacency(vertices, faces):
    """Build a sparse weighted adjacency matrix for the mesh."""
    num_vertices = len(vertices)
    adjacency_matrix = lil_matrix((num_vertices, num_vertices), dtype=np.float32)
    for face in tqdm(faces, desc="Building sparse adjacency matrix", leave=False):
        for i in range(3):
            for j in range(i + 1, 3):
                vi, vj = face[i], face[j]
                dist = np.linalg.norm(vertices[vi] - vertices[vj])
                adjacency_matrix[vi, vj] = dist
                adjacency_matrix[vj, vi] = dist
    return adjacency_matrix.tocsr()

def fast_dijkstra_sparse(adjacency_matrix, start_vertex):
    """Optimized Dijkstra's algorithm using a sparse adjacency matrix."""
    num_vertices = adjacency_matrix.shape[0]
    distances = np.full(num_vertices, np.inf)
    distances[start_vertex] = 0
    pq = [(0, start_vertex)]  # min-heap
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue
        row = adjacency_matrix[current_vertex]
        for neighbor, edge_weight in zip(row.indices, row.data):
            new_distance = current_distance + edge_weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))
    return torch.tensor(distances, dtype=torch.float32, device=device)

def compute_three_pole_representation(vertices, faces, ref1, ref2, ref3):
    """
    Compute geodesic distances from three reference points and return:
      - sum_potential = d1 + d2 + d3
      - diff_potential = |d1 - d2 - d3|
    """
    st.info("Computing geodesic distances (this may take a moment)...")
    start_time = time.time()
    adjacency_matrix = build_sparse_adjacency(vertices, faces)
    d1 = fast_dijkstra_sparse(adjacency_matrix, ref1)
    d2 = fast_dijkstra_sparse(adjacency_matrix, ref2)
    d3 = fast_dijkstra_sparse(adjacency_matrix, ref3)
    sum_potential = d1 + d2 + d3
    diff_potential = torch.abs(d1 - d2 - d3)
    st.write(f"Sum potential range: {float(sum_potential.min())} to {float(sum_potential.max())}")
    st.write(f"Diff potential range: {float(diff_potential.min())} to {float(diff_potential.max())}")
    st.write(f"Geodesic computations finished in {time.time()-start_time:.2f} seconds")
    return sum_potential.cpu(), diff_potential.cpu()

# ---------------- Landmark and Reference Functions --------------------------

def read_bnd_file(file_bytes):
    """
    Read a .bnd file (landmarks) and return an (N,3) numpy array.
    Expected format: index, X, Y, Z (tab- or space-delimited).
    """
    try:
        text = file_bytes.decode("utf-8")
        s = StringIO(text)
        data = np.genfromtxt(s, delimiter=None)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] == 4:
            landmarks = data[:, 1:4]
        elif data.shape[1] >= 3:
            landmarks = data[:, :3]
        else:
            raise ValueError("Unrecognized .bnd file format.")
        return landmarks
    except Exception as e:
        st.error(f"Error reading .bnd file: {e}")
        return None

def find_closest_vertex(mesh, point):
    """
    Find the index of the vertex in mesh.vertices closest to the given point.
    """
    vertices = np.array(mesh.vertices)
    distances = np.linalg.norm(vertices - point, axis=1)
    return int(np.argmin(distances))

def compute_nose_top_index(mesh, x_threshold=0.2):
    """
    Compute a simple curvature approximation for vertices in the central region.
    The search is restricted to vertices whose x and y coordinates are near the mesh center.
    Returns the index of the vertex with the highest discrete curvature as an approximation for the nose top.
    """
    vertices = np.array(mesh.vertices)
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    mask = (np.abs(vertices[:, 0] - center[0]) < x_threshold) & (np.abs(vertices[:, 1] - center[1]) < x_threshold)
    curvature = np.zeros(len(vertices))
    for i in range(len(vertices)):
        if not mask[i]:
            continue
        neighbors = mesh.vertex_neighbors[i]
        if len(neighbors) > 0:
            neighbor_positions = vertices[neighbors]
            curvature[i] = np.linalg.norm(vertices[i] - neighbor_positions.mean(axis=0))
    nose_index = int(np.argmax(curvature))
    return nose_index

# -------------------- Visualization Functions -------------------------------

def plot_epsilon_variation(targets, epsilons, title):
    """
    Plot the epsilon variation function (target vs. epsilon) using Plotly.
    The x-axis is labeled as 'Potential Value (P)' with a ruler to indicate scale.
    """
    eps_func = interp1d(targets, epsilons, kind='linear', fill_value="extrapolate")
    p_sample = np.linspace(np.min(targets), np.max(targets), 100)
    eps_values = eps_func(p_sample)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_sample, y=eps_values, mode='lines', name='ε = f(P)'))
    fig.add_trace(go.Scatter(x=targets, y=epsilons, mode='markers', name='Sampled ε'))
    fig.update_layout(title=title,
                      xaxis_title="Potential Value (P)",
                      yaxis_title="Tolerance (ε)",
                      xaxis=dict(tickformat=".2f"))
    return fig

def plot_xy_potential(xy_coords, potential_vals, title="Potential Values in the XY Plane"):
    """
    Create a scatter plot of the potential values on the fixed z-slice.
    The x- and y-axes show spatial coordinates and the color axis indicates the potential value.
    """
    fig = go.Figure(data=go.Scatter(
        x=xy_coords[:, 0],
        y=xy_coords[:, 1],
        mode='markers',
        marker=dict(color=potential_vals, colorscale='Viridis', colorbar=dict(title="Potential (P)"), size=4)
    ))
    fig.update_layout(title=title,
                      xaxis_title="X Coordinate",
                      yaxis_title="Y Coordinate")
    return fig

def plot_mesh_contours(mesh, vertices, faces, xy_coords, potential_vals, targets, tol, contour_name, colors):
    """
    Plot contours on the mesh by selecting points (in the xy plane) where the potential
    is within tol of the target values.
    """
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.5, color='lightblue', name='Mesh'
    ))
    for i, target in enumerate(targets):
        indices = np.where(np.abs(potential_vals - target) < tol)[0]
        if len(indices) > 0:
            pts = xy_coords[indices]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers', marker=dict(color=colors[i % len(colors)], size=3),
                name=f"{contour_name} Contour {i+1}"
            ))
    return fig

# -------------------- Device and Streamlit App ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write("CUDA available:", torch.cuda.is_available())

st.title("Raffinement d'epsilon in Three-Pole Geodesic Representation")

# File upload: landmarks (.bnd) and mesh (.obj)
uploaded_bnd = st.file_uploader("Upload landmarks (.bnd)", type=["bnd"])
uploaded_obj = st.file_uploader("Upload face mesh (.obj)", type=["obj"])

if uploaded_bnd is not None and uploaded_obj is not None:
    landmarks = read_bnd_file(uploaded_bnd.getvalue())
    if landmarks is None:
        st.stop()
    st.write(f"Extracted {landmarks.shape[0]} landmarks from .bnd file.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp:
        tmp.write(uploaded_obj.getvalue())
        mesh_path = tmp.name
    try:
        vertices, faces, mesh = load_obj_mesh(mesh_path)
        st.success("Mesh loaded successfully!")
    except Exception as e:
        st.error(f"Error loading mesh: {e}")
        st.stop()
    
    # Display mesh and landmarks for inspection
    fig_landmarks = go.Figure()
    fig_landmarks.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.5, color='lightblue', name='Mesh'
    ))
    fig_landmarks.add_trace(go.Scatter3d(
        x=landmarks[:, 0], y=landmarks[:, 1], z=landmarks[:, 2],
        mode='markers', marker=dict(color='red', size=5),
        name='Landmarks'
    ))
    st.plotly_chart(fig_landmarks, use_container_width=True)
    
    # Get reference points: inner eyes (landmarks at indexes 0 and 8)
    if landmarks.shape[0] < 9:
        st.error("Not enough landmarks found. Expecting at least 9 landmarks.")
        st.stop()
    left_eye_pt = landmarks[0]
    right_eye_pt = landmarks[8]
    left_eye_index = find_closest_vertex(mesh, left_eye_pt)
    right_eye_index = find_closest_vertex(mesh, right_eye_pt)
    nose_index = compute_nose_top_index(mesh)
    st.write(f"Reference vertex indices: Left Eye = {left_eye_index}, Right Eye = {right_eye_index}, Nose Top = {nose_index}")
    
    # Compute three-pole geodesic distances
    sum_pot, diff_pot = compute_three_pole_representation(vertices, faces, left_eye_index, right_eye_index, nose_index)
    
    # Work on a fixed z–plane (front view)
    vertices_np = vertices
    z_values = vertices_np[:, 2]
    fixed_z = np.median(z_values)
    z_tol = st.slider("Select z–slice tolerance", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    plane_mask = np.abs(z_values - fixed_z) < z_tol
    xy_vertices = vertices_np[plane_mask]
    sum_pot_xy = sum_pot.numpy()[plane_mask]
    diff_pot_xy = diff_pot.numpy()[plane_mask]
    st.write(f"Using {np.sum(plane_mask)} vertices in the fixed z–plane (z ~ {fixed_z:.3f}).")
    
    # Visualize the sampling of potential values on the xy plane
    fig_xy = plot_xy_potential(xy_vertices, sum_pot_xy, title="Sampling of Sum Potential in the XY Plane")
    st.plotly_chart(fig_xy, use_container_width=True)
    
    # Define the percentile slider for ε computation.
    # For each target potential value, we compute the tolerance as the chosen percentile
    # of |P(x,y)-target|. This defines our f(P)=ε function.
    perc = st.slider("Select percentile for ε computation", min_value=1, max_value=50, value=10, step=1)
    
    # --- Dense sampling for a smooth ε variation function ---
    num_levels_dense = 120
    targets_sum_dense = np.linspace(np.min(sum_pot_xy), np.max(sum_pot_xy), num_levels_dense)
    epsilons_sum_dense = [np.percentile(np.abs(sum_pot_xy - t), perc) for t in targets_sum_dense]

    targets_diff_dense = np.linspace(np.min(diff_pot_xy), np.max(diff_pot_xy), num_levels_dense)
    epsilons_diff_dense = [np.percentile(np.abs(diff_pot_xy - t), perc) for t in targets_diff_dense]

    fig_eps_sum_dense = plot_epsilon_variation(targets_sum_dense, epsilons_sum_dense, "Dense ε Variation for Sum Potential")
    st.plotly_chart(fig_eps_sum_dense, use_container_width=True)

    fig_eps_diff_dense = plot_epsilon_variation(targets_diff_dense, epsilons_diff_dense, "Dense ε Variation for Diff Potential")
    st.plotly_chart(fig_eps_diff_dense, use_container_width=True)
    
    # --- Coarse targets for displaying mesh contours (to avoid clutter) ---
    num_levels_coarse = 5
    targets_sum_vis = np.quantile(sum_pot_xy, np.linspace(0.1, 0.9, num_levels_coarse))
    targets_diff_vis = np.quantile(diff_pot_xy, np.linspace(0.1, 0.9, num_levels_coarse))
    
    tol_sum = st.slider("Tolerance for Sum Potential contours", min_value=0.001, max_value=0.5, value=0.05, step=0.001)
    tol_diff = st.slider("Tolerance for Diff Potential contours", min_value=0.001, max_value=0.5, value=0.05, step=0.001)
    
    # Prepare 3D plot showing the mesh with contour points
    mesh_fig = go.Figure()
    mesh_fig.add_trace(go.Mesh3d(
        x=vertices_np[:, 0], y=vertices_np[:, 1], z=vertices_np[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.5, color='lightblue', name='Mesh'
    ))
    bright_colors = ['orange', 'red', 'blue', 'green', 'yellow']
    for i, target in enumerate(targets_sum_vis):
        idx = np.where(np.abs(sum_pot_xy - target) < tol_sum)[0]
        if len(idx) > 0:
            pts = xy_vertices[idx]
            mesh_fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers', marker=dict(color=bright_colors[i % len(bright_colors)], size=3),
                name=f"Sum Contour {i+1}"
            ))
    dark_colors = ['brown', 'black', 'darkgray', 'dimgray', 'darkred']
    for i, target in enumerate(targets_diff_vis):
        idx = np.where(np.abs(diff_pot_xy - target) < tol_diff)[0]
        if len(idx) > 0:
            pts = xy_vertices[idx]
            mesh_fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers', marker=dict(color=dark_colors[i % len(dark_colors)], size=3),
                name=f"Diff Contour {i+1}"
            ))
    ref_colors = {'Left Eye': 'green', 'Right Eye': 'orange', 'Nose Top': 'red'}
    ref_indices = {'Left Eye': left_eye_index, 'Right Eye': right_eye_index, 'Nose Top': nose_index}
    for label, idx in ref_indices.items():
        pt = mesh.vertices[idx]
        mesh_fig.add_trace(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]],
            mode='markers', marker=dict(color=ref_colors[label], size=8),
            name=label
        ))
    mesh_fig.update_layout(title="3D Mesh with Contours and Reference Points", scene=dict(aspectmode='data'))
    st.plotly_chart(mesh_fig, use_container_width=True)
