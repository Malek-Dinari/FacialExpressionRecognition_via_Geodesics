#!/usr/bin/env python
# bipolar_cuda.py
# Script pour calculer la représentation bipolaire des potentiels géodésiques sur un maillage 3D
# Utilise CUDA si disponible
# Auteur/Maintener : Malek DINARI

import trimesh
import numpy as np
import torch
import plotly.graph_objs as go
import os
import time
import heapq
from tqdm import tqdm
from scipy.sparse import lil_matrix

# Définir l'appareil CUDA si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Appareil utilisé :", device)

# Définir deux indices de référence pour la représentation bipolaire
seed_index1 = 22340
seed_index2 = 22947

def load_obj_mesh(filepath):
    """
    Charger un fichier OBJ et extraire les sommets et les faces du maillage.
    """
    mesh = trimesh.load(filepath, process=True)
    return np.array(mesh.vertices), np.array(mesh.faces), mesh

def build_sparse_adjacency(vertices, faces):
    """
    Construire une matrice d'adjacence creuse pondérée pour le maillage.
    """
    num_vertices = len(vertices)
    adjacency_matrix = lil_matrix((num_vertices, num_vertices), dtype=np.float32)
    for face in tqdm(faces, desc="Construction de la matrice d'adjacence"):
        for i in range(3):
            for j in range(i + 1, 3):
                vi, vj = face[i], face[j]
                dist = np.linalg.norm(vertices[vi] - vertices[vj])
                adjacency_matrix[vi, vj] = dist
                adjacency_matrix[vj, vi] = dist
    return adjacency_matrix.tocsr()

def fast_dijkstra_sparse(adjacency_matrix, start_vertex):
    """
    Algorithme de Dijkstra optimisé utilisant une matrice d'adjacence creuse.
    """
    num_vertices = adjacency_matrix.shape[0]
    distances = np.full(num_vertices, np.inf)
    distances[start_vertex] = 0
    # File de priorité (min-heap)
    pq = [(0, start_vertex)]
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue
        
        # Parcours des voisins via la matrice creuse
        for neighbor, edge_weight in zip(adjacency_matrix[current_vertex].indices,
                                         adjacency_matrix[current_vertex].data):
            new_distance = current_distance + edge_weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))
    
    return torch.tensor(distances, dtype=torch.float32, device=device)

def compute_bipolar_representation(vertices, faces, ref1, ref2):
    """
    Calculer les distances géodésiques à partir de deux points de référence et
    générer la représentation bipolaire.
    
    La représentation bipolaire se compose de :
      - somme_des_potentiels = d1 + d2
      - difference_des_potentiels = |d1 - d2|
    """
    print("[INFO] Calcul des distances géodésiques avec CUDA...")
    start_time = time.time()
    adjacency_matrix = build_sparse_adjacency(vertices, faces)
    d1 = fast_dijkstra_sparse(adjacency_matrix, ref1).to(device)
    d2 = fast_dijkstra_sparse(adjacency_matrix, ref2).to(device)
    
    somme_des_potentiels = d1 + d2
    difference_des_potentiels = torch.abs(d1 - d2)
    
    print(f"[INFO] Plage de 'somme_des_potentiels': {torch.min(somme_des_potentiels).item()} à {torch.max(somme_des_potentiels).item()}")
    print(f"[INFO] Plage de 'difference_des_potentiels': {torch.min(difference_des_potentiels).item()} à {torch.max(difference_des_potentiels).item()}")
    print(f"[INFO] Calcul terminé en {time.time() - start_time:.2f} secondes")
    
    return somme_des_potentiels.cpu(), difference_des_potentiels.cpu()

def compute_quantile_distances(distances, num_contours=7):
    """
    Calculer des distances cibles sous forme de quantiles de la distribution de distances.
    """
    quantiles = np.linspace(0.1, 0.9, num_contours)
    target_distances = np.quantile(distances, quantiles)
    return target_distances.tolist()

def plot_bipolar_contours(mesh, somme_des_potentiels, difference_des_potentiels,
                           target_distances_somme, target_distances_diff, tolerance,
                           show_somme=True, show_diff=True):
    """
    Afficher les contours géodésiques pour la représentation bipolaire.
    """
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    fig = go.Figure()
    
    # Affichage de la surface du maillage
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        color='gray',
        opacity=0.5,
        name="Maillage"
    ))
    
    if show_somme and target_distances_somme is not None:
        couleurs_somme = ['orange', 'red', 'blue', 'green', 'yellow', 'purple', 'cyan']
        for i, target in enumerate(target_distances_somme):
            indices = np.where(np.isclose(somme_des_potentiels.numpy(), target, atol=tolerance))[0]
            vertices_sel = mesh.vertices[indices]
            fig.add_trace(go.Scatter3d(
                x=vertices_sel[:, 0], y=vertices_sel[:, 1], z=vertices_sel[:, 2],
                mode='markers',
                marker=dict(color=couleurs_somme[i % len(couleurs_somme)], size=2),
                name=f"Contour Somme {i+1}"
            ))
            print(f"Contour Somme {i+1} : {len(indices)} points pour cible {target} avec tolérance {tolerance}")
    
    if show_diff and target_distances_diff is not None:
        couleurs_diff = ['brown', 'black', 'darkgray', 'dimgray', 'darkred', 'darkblue', 'darkgreen']
        for i, target in enumerate(target_distances_diff):
            indices = np.where(np.isclose(difference_des_potentiels.numpy(), target, atol=tolerance))[0]
            vertices_sel = mesh.vertices[indices]
            fig.add_trace(go.Scatter3d(
                x=vertices_sel[:, 0], y=vertices_sel[:, 1], z=vertices_sel[:, 2],
                mode='markers',
                marker=dict(color=couleurs_diff[i % len(couleurs_diff)], size=2),
                name=f"Contour Différence {i+1}"
            ))
            print(f"Contour Différence {i+1} : {len(indices)} points pour cible {target} avec tolérance {tolerance}")
    
    # Ajout des points de référence
    ref_points = [seed_index1, seed_index2]
    couleurs_ref = ['red', 'blue']
    for i, ref in enumerate(ref_points):
        fig.add_trace(go.Scatter3d(
            x=[mesh.vertices[ref][0]], y=[mesh.vertices[ref][1]], z=[mesh.vertices[ref][2]],
            mode='markers',
            marker=dict(color=couleurs_ref[i], size=6),
            name=f"Point de référence {i+1}"
        ))
    
    # Configuration finale de la figure
    fig.update_layout(
        title="Représentation Bipolaire des Potentiels Géodésiques",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        showlegend=True
    )
    
    fig.show()

def process_mesh(filepath, tolerance=0.25, show_somme=True, show_diff=True, custom_target_distances=None):
    """
    Traiter le maillage : calculer les distances géodésiques, afficher les contours, et sauvegarder les résultats.
    """
    print(f"[INFO] Traitement de {filepath}...")
    vertices, faces, mesh = load_obj_mesh(filepath)
    somme_des_potentiels, difference_des_potentiels = compute_bipolar_representation(vertices, faces, seed_index1, seed_index2)
    
    # Calcul ou récupération des distances cibles
    if custom_target_distances is not None:
        target_distances_somme = custom_target_distances.get('somme', None)
        target_distances_diff = custom_target_distances.get('diff', None)
    else:
        target_distances_somme = compute_quantile_distances(somme_des_potentiels.numpy())
        target_distances_diff = compute_quantile_distances(difference_des_potentiels.numpy())
    
    # Sauvegarder les résultats calculés
    mesh_basename = os.path.basename(filepath).replace('.obj', '')
    np.save(f'somme_des_potentiels-{mesh_basename}.npy', somme_des_potentiels.numpy())
    np.save(f'difference_des_potentiels-{mesh_basename}.npy', difference_des_potentiels.numpy())
    print(f"[INFO] Résultats sauvegardés sous somme_des_potentiels-{mesh_basename}.npy et difference_des_potentiels-{mesh_basename}.npy")
    
    # Affichage des résultats
    plot_bipolar_contours(mesh, somme_des_potentiels, difference_des_potentiels,
                           target_distances_somme, target_distances_diff, tolerance,
                           show_somme, show_diff)

if __name__ == "__main__":
    # Modifier le chemin vers le fichier maillage selon vos besoins
    mesh_path = r"C:\Users\Lenovo\Desktop\programming\TASKS-and-PROJECTS-2024-25\REC-VISAGE-4D\data\deformed_F0001_AN01WH_F3Dsur.obj"
    # Conversion du chemin pour WSL2 (si nécessaire)
    wsl_path = mesh_path.replace("\\", "/").replace("C:", "/mnt/c")
    
    # Exemple d'utilisation avec des distances cibles personnalisées et une tolérance définie
    custom_targets = {
        'somme': [97, 108, 119, 130, 141, 152, 163, 174, 185, 196, 207, 218, 229],
        'diff': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    }
    
    process_mesh(wsl_path, tolerance=0.25, show_somme=True, show_diff=True, custom_target_distances=custom_targets)
