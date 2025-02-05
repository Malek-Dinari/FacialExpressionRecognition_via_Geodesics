import trimesh
import numpy as np
import torch
import plotly.graph_objs as go
import os
import time
import heapq
from tqdm import tqdm
from scipy.sparse import lil_matrix

# Set CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# Define three reference indices
seed_index1 = 13714
seed_index2 = 22947
seed_index3 = 22340

def load_obj_mesh(filepath):
    """
    Load an OBJ mesh file and extract vertices and faces.
    """
    mesh = trimesh.load(filepath, process=True)
    return np.array(mesh.vertices), np.array(mesh.faces), mesh


def build_sparse_adjacency(vertices, faces):
    """
    Build a sparse weighted adjacency matrix for the mesh.
    """
    num_vertices = len(vertices)
    adjacency_matrix = lil_matrix((num_vertices, num_vertices), dtype=np.float32)
    for face in tqdm(faces, desc="Building sparse adjacency matrix"):
        for i in range(3):
            for j in range(i + 1, 3):
                vi, vj = face[i], face[j]
                dist = np.linalg.norm(vertices[vi] - vertices[vj])
                adjacency_matrix[vi, vj] = dist
                adjacency_matrix[vj, vi] = dist
    return adjacency_matrix.tocsr()  # Convert to compressed sparse row format

def build_adjacency_matrix(vertices, faces):
    """
    Build a weighted adjacency matrix for the mesh.
    """
    num_vertices = len(vertices)
    adjacency_matrix = np.zeros((num_vertices, num_vertices))

    for face in tqdm(faces, desc="Building adjacency matrix"):
        for i in range(3):
            for j in range(i + 1, 3):
                vi, vj = face[i], face[j]
                dist = np.linalg.norm(vertices[vi] - vertices[vj])
                adjacency_matrix[vi, vj] = dist
                adjacency_matrix[vj, vi] = dist

    return adjacency_matrix

def fast_dijkstra_sparse(adjacency_matrix, start_vertex):
    """
    Optimized Dijkstra's algorithm using a sparse adjacency matrix.
    """
    num_vertices = adjacency_matrix.shape[0]
    distances = np.full(num_vertices, np.inf)
    distances[start_vertex] = 0
    pq = [(0, start_vertex)]  # Min-heap priority queue
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue
        
        # Iterate over neighbors using the sparse matrix
        for neighbor, edge_weight in zip(adjacency_matrix[current_vertex].indices, adjacency_matrix[current_vertex].data):
            new_distance = current_distance + edge_weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))
    
    return torch.tensor(distances, dtype=torch.float32, device=device)

def compute_three_pole_representation(vertices, faces, ref1, ref2, ref3):
    """
    Compute geodesic distances from three reference points.
    """
    print("[INFO] Computing geodesic distances using CUDA...")
    start_time = time.time()
    # Build sparse adjacency matrix
    adjacency_matrix = build_sparse_adjacency(vertices, faces)
    # Compute distances
    d1 = fast_dijkstra_sparse(adjacency_matrix, ref1).to(device)
    d2 = fast_dijkstra_sparse(adjacency_matrix, ref2).to(device)
    d3 = fast_dijkstra_sparse(adjacency_matrix, ref3).to(device)
    sum_distances = d1 + d2 + d3
    alternating_sum_distances = torch.abs(d1 - d2 + d3)
    print(f"[INFO] Sum distances range: {torch.min(sum_distances).item()} to {torch.max(sum_distances).item()}")
    print(f"[INFO] Alternating sum distances range: {torch.min(alternating_sum_distances).item()} to {torch.max(alternating_sum_distances).item()}")
    print(f"[INFO] Computation finished in {time.time() - start_time:.2f} seconds")
    return sum_distances.cpu(), alternating_sum_distances.cpu()

def compute_quantile_distances(distances, num_contours=7):
    """
    Compute target distances as quantiles of the distance distribution.
    """
    quantiles = np.linspace(0.1, 0.9, num_contours)  # Avoid extreme values
    target_distances = np.quantile(distances.cpu().numpy(), quantiles)
    return target_distances.tolist()

def plot_tripolar_contours(mesh, sum_distances, alternating_sum_distances, target_distances_sum, target_distances_diff, tolerance, show_sum=True, show_diff=True):
    """
    Plot tripolar geodesic contours for sum and alternating sum distances.
    """
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    fig = go.Figure()
    
    # Mesh surface
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        color='gray',
        opacity=0.5
    ))
    
    if show_sum and target_distances_sum is not None:
        # Contour plot for sum distances (bright colors)
        bright_colors = ['orange', 'red', 'blue', 'green', 'yellow', 'purple', 'cyan']
        for i, target in enumerate(target_distances_sum):
            selected_indices = np.where(np.isclose(sum_distances.numpy(), target, atol=tolerance))[0]
            print(f"Sum contour {i+1}: {len(selected_indices)} points found for target {target} with tolerance {tolerance}")
            selected_vertices = mesh.vertices[selected_indices]
            fig.add_trace(go.Scatter3d(
                x=selected_vertices[:, 0], y=selected_vertices[:, 1], z=selected_vertices[:, 2],
                mode='markers',
                marker=dict(color=bright_colors[i % len(bright_colors)], size=2),
                name=f"Sum Contour {i+1}",
                visible=True  # Allow toggling from legend
            ))
    
    if show_diff and target_distances_diff is not None:
        # Contour plot for alternating sum distances (dark colors)
        dark_colors = ['brown', 'black', 'darkgray', 'dimgray', 'darkred', 'darkblue', 'darkgreen']
        for i, target in enumerate(target_distances_diff):
            selected_indices = np.where(np.isclose(alternating_sum_distances.numpy(), target, atol=tolerance))[0]
            print(f"Difference contour {i+1}: {len(selected_indices)} points found for target {target} with tolerance {tolerance}")
            selected_vertices = mesh.vertices[selected_indices]
            fig.add_trace(go.Scatter3d(
                x=selected_vertices[:, 0], y=selected_vertices[:, 1], z=selected_vertices[:, 2],
                mode='markers',
                marker=dict(color=dark_colors[i % len(dark_colors)], size=2),
                name=f"Difference Contour {i+1}",
                visible=True  # Allow toggling from legend
            ))
    
    # Add reference points as small spheres
    ref_points = [seed_index1, seed_index2, seed_index3]
    colors = ['red', 'green', 'blue']
    for i, ref in enumerate(ref_points):
        fig.add_trace(go.Scatter3d(
            x=[mesh.vertices[ref][0]], y=[mesh.vertices[ref][1]], z=[mesh.vertices[ref][2]],
            mode='markers',
            marker=dict(color=colors[i], size=6),
            name=f"Reference Point {i+1}"
        ))
    
    # Update layout
    fig.update_layout(
        title="Tripolar Geodesic Representation",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        showlegend=True
    )
    
    fig.show()

def process_mesh(filepath, tolerance=0.01, show_sum=True, show_diff=True, custom_target_distances=None):
    """
    Process the mesh file: Compute tripolar distances, plot contours, and save results.
    """
    print(f"[INFO] Processing {filepath}...")
    # Load mesh
    vertices, faces, mesh = load_obj_mesh(filepath)
    # Compute sum and alternating sum of geodesic distances
    sum_distances, alternating_sum_distances = compute_three_pole_representation(vertices, faces, seed_index1, seed_index2, seed_index3)
    # Dynamically compute target distances or use custom ones
    if custom_target_distances is not None:
        target_distances_sum = custom_target_distances.get('sum', None)
        target_distances_diff = custom_target_distances.get('diff', None)
    else:
        target_distances_sum = compute_quantile_distances(sum_distances)
        target_distances_diff = compute_quantile_distances(alternating_sum_distances)
    # Save computed values
    mesh_basename = os.path.basename(filepath).replace('.obj', '')
    np.save(f'sum_distances-{mesh_basename}.npy', sum_distances.numpy())
    np.save(f'alternating_sum_distances-{mesh_basename}.npy', alternating_sum_distances.numpy())
    print(f"[INFO] Results saved as sum_distances-{mesh_basename}.npy and alternating_sum_distances-{mesh_basename}.npy")
    # Plot results
    plot_tripolar_contours(mesh, sum_distances, alternating_sum_distances, target_distances_sum, target_distances_diff, tolerance, show_sum, show_diff)

if __name__ == "__main__":
    # Change the mesh path as needed
    mesh_path = r"C:\Users\Lenovo\Desktop\programming\TASKS-and-PROJECTS-2024-25\REC-VISAGE-4D\data\F0001_AN01WH_F3Dsur.obj"
    wsl_path = mesh_path.replace("\\", "/").replace("C:", "/mnt/c")

    # Example usage with custom target distances and tolerance
    custom_targets = {
        'sum': [104, 119, 129, 139, 149, 159, 174],
        'diff': [50, 60, 70, 80, 90, 100, 110]
    }
    process_mesh(wsl_path, tolerance=2.0, show_sum=True, show_diff=True, custom_target_distances=custom_targets)