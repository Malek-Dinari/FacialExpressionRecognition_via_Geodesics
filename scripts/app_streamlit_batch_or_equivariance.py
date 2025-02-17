#!/usr/bin/env python
"""
app_streamlit_batch_or_equivariance.py
"""














import streamlit as st
import numpy as np
import trimesh
import torch
import heapq
from scipy.sparse import lil_matrix
import tempfile
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from numpy.linalg import norm, svd
from io import BytesIO, StringIO
import os

###########################################
# Fonctions de base : chargement, repères, potentiels, etc.
###########################################

def read_bnd_file(file_bytes):
    """Lire le fichier .bnd et extraire les repères sous forme d'un tableau (N, 3)."""
    try:
        text = file_bytes.decode("utf-8")
    except Exception as e:
        st.error(f"Erreur lors du décodage du fichier .bnd: {e}")
        return None
    lines = text.splitlines()
    st.write("Extrait des premières lignes du fichier .bnd :")
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

def load_mesh(filepath):
    """Charger un maillage 3D avec trimesh."""
    mesh = trimesh.load(filepath, process=True)
    return mesh

def build_sparse_adjacency(vertices, faces):
    """Construire une matrice d'adjacence creuse pondérée pour le maillage."""
    num_vertices = len(vertices)
    adjacency_matrix = lil_matrix((num_vertices, num_vertices), dtype=np.float32)
    for face in faces:
        for i in range(3):
            for j in range(i+1, 3):
                vi, vj = face[i], face[j]
                d = norm(vertices[vi] - vertices[vj])
                adjacency_matrix[vi, vj] = d
                adjacency_matrix[vj, vi] = d
    return adjacency_matrix.tocsr()

def fast_dijkstra_sparse(adjacency_matrix, start_vertex):
    """Exécuter Dijkstra sur la matrice d'adjacence."""
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
    return torch.tensor(distances, dtype=torch.float32)

def compute_bipolar_representation(vertices, faces, ref_idx1, ref_idx2):
    """
    Calculer les potentiels géodésiques d1 et d2 depuis deux repères (yeux intérieurs)
    et retourner la somme (d1+d2) et la différence absolue (|d1-d2|).
    """
    st.write("[INFO] Calcul des potentiels géodésiques...")
    adj = build_sparse_adjacency(vertices, faces)
    d1 = fast_dijkstra_sparse(adj, ref_idx1)
    d2 = fast_dijkstra_sparse(adj, ref_idx2)
    somme = d1 + d2
    diff = torch.abs(d1 - d2)
    st.write(f"[INFO] Plage somme: {torch.min(somme).item()} à {torch.max(somme).item()}")
    st.write(f"[INFO] Plage diff: {torch.min(diff).item()} à {torch.max(diff).item()}")
    return somme.cpu(), diff.cpu()

def find_nearest_vertex(vertices, point):
    """Trouver l'indice du vertex le plus proche d'un point donné."""
    dists = norm(vertices - point, axis=1)
    return np.argmin(dists)

###########################################
# Réordonnancement et réparamétrisation (B-spline avec fermeture forcée)
###########################################

def order_contour_points(points, contour_type="sum"):
    """
    Réordonner les points d'un contour.
    - Pour 'sum', on utilise un ordre basé sur les extrémités.
    - Pour 'diff', on sépare en points à gauche et à droite et on ordonne séparément.
    """
    if len(points) <= 1:
        return points.copy()
    if contour_type == "sum":
        dist_matrix = norm(points[:, None] - points, axis=2)
        i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        ordered = []
        remaining = set(range(len(points)))
        current = i
        ordered.append(current)
        remaining.remove(current)
        while remaining:
            nearest = min(remaining, key=lambda x: norm(points[current] - points[x]))
            ordered.append(nearest)
            remaining.remove(nearest)
            current = nearest
        if ordered[-1] != j:
            ordered = ordered[::-1]
        return points[ordered]
    elif contour_type == "diff":
        centroid = np.mean(points, axis=0)
        left_points = points[points[:, 0] < centroid[0]]
        right_points = points[points[:, 0] >= centroid[0]]
        left_ordered = left_points[np.argsort(left_points[:, 1])]
        right_ordered = right_points[np.argsort(right_points[:, 1])]
        return np.vstack((left_ordered, right_ordered[::-1]))
    else:
        raise ValueError("Contour type invalide. Utiliser 'sum' ou 'diff'.")

def fit_bspline_curve(points, n_samples=50, s=0, contour_type="sum", force_closed=False):
    """
    Ajuster une B-spline sur un contour réordonné et le rééchantillonner par abscisse curviligne.
    Si force_closed est True, le contour est fermé en ajoutant le premier point à la fin (si nécessaire).
    Retourne la courbe interpolée, le tuple (tck) et les valeurs de paramètre u_new.
    """
    if len(points) < 4:
        return points, None, None
    points_ordered = order_contour_points(points, contour_type=contour_type)
    if force_closed:
        if norm(points_ordered[0] - points_ordered[-1]) > 1e-3:
            points_ordered = np.vstack([points_ordered, points_ordered[0]])
        is_closed = True
    else:
        is_closed = (contour_type == "sum") and (norm(points_ordered[0] - points_ordered[-1]) < 1e-3)
    k = 3 if len(points_ordered) >= 4 else min(3, len(points_ordered)-1)
    try:
        dists = np.linalg.norm(np.diff(points_ordered, axis=0), axis=1)
        arc_length = np.concatenate(([0], np.cumsum(dists)))
        arc_length_norm = arc_length / arc_length[-1] if arc_length[-1] != 0 else np.linspace(0, 1, len(points_ordered))
        tck, u = splprep(points_ordered.T, u=arc_length_norm, s=s, k=k, per=is_closed)
        u_new = np.linspace(0, 1, n_samples)
        if is_closed:
            u_new = u_new[:-1]
        curve = np.array(splev(u_new, tck)).T
        return curve, tck, u_new
    except Exception as e:
        st.error(f"B-spline error: {str(e)}")
        return points_ordered, None, None

def compute_tangent(tck, u_val):
    """
    Calculer le vecteur tangent à la courbe B-spline à la valeur de paramètre u_val.
    Utilise splev avec der=1 pour obtenir la dérivée.
    """
    deriv = splev(u_val, tck, der=1)
    tangent = np.array(deriv)
    if norm(tangent) != 0:
        tangent = tangent / norm(tangent)
    return tangent

###########################################
# Extraction des intersections entre segments (pour obtenir 4 intersections)
###########################################

def segment_intersection(P, Q, R, S, tol=0.01):
    """
    Calculer l'intersection entre les segments [P, Q] et [R, S] dans R^3.
    On résout pour t et s dans les équations paramétriques.
    Si 0<=t<=1 et 0<=s<=1 et la distance entre les deux points est < tol,
    renvoie la moyenne des points d'intersection.
    """
    d = Q - P
    e = S - R
    denom = (np.dot(d, d) * np.dot(e, e)) - (np.dot(d, e))**2
    if np.abs(denom) < 1e-8:
        return None  # segments presque parallèles
    t = (np.dot(R - P, d)*np.dot(e, e) - np.dot(R - P, e)*np.dot(d, e)) / denom
    s_param = (np.dot(R - P, d) + t*np.dot(d, e)) / np.dot(e, e)
    if 0 <= t <= 1 and 0 <= s_param <= 1:
        point1 = P + t*d
        point2 = R + s_param*e
        if norm(point1 - point2) < tol:
            return (point1 + point2) / 2.0
    return None

def find_all_intersections_segments(curve1, curve2, tol=0.5):
    """
    Pour chaque segment de curve1 et chaque segment de curve2, calculer l'intersection.
    Regrouper ensuite les points proches pour éviter les doublons.
    """
    candidates = []
    for i in range(len(curve1)-1):
        P = curve1[i]
        Q = curve1[i+1]
        for j in range(len(curve2)-1):
            R = curve2[j]
            S = curve2[j+1]
            ip = segment_intersection(P, Q, R, S, tol=tol)
            if ip is not None:
                candidates.append(ip)
    unique_ips = []
    for pt in candidates:
        if not unique_ips:
            unique_ips.append(pt)
        else:
            if all(norm(pt - np.array(up)) >= tol/2 for up in unique_ips):
                unique_ips.append(pt)
            else:
                # Moyenne des points proches
                for k, up in enumerate(unique_ips):
                    if norm(pt - np.array(up)) < tol/2:
                        unique_ips[k] = (np.array(up) + pt)/2.0
    return np.array(unique_ips)

def find_four_intersections(curve1, curve2, tol=0.5):
    """
    Extraire tous les points d'intersection et sélectionner 4 points représentatifs.
    Si plus de 4 intersections sont trouvées, trier par angle autour du centroïde et sélectionner 4 points équidistants.
    """
    intersections = find_all_intersections_segments(curve1, curve2, tol=tol)
    if intersections.size == 0:
        return intersections
    if len(intersections) > 4:
        centroid = np.mean(intersections, axis=0)
        angles = np.arctan2(intersections[:,1]-centroid[1], intersections[:,0]-centroid[0])
        sorted_indices = np.argsort(angles)
        sorted_ips = intersections[sorted_indices]
        indices = np.linspace(0, len(sorted_ips)-1, 4, dtype=int)
        return sorted_ips[indices]
    else:
        return intersections

###########################################
# Construction du descripteur S(u,v)
###########################################

def get_Suv_descriptor(intersections, vertices, sums, diffs):
    """
    Pour chaque point d'intersection (3D), trouver le vertex le plus proche et récupérer
    ses potentiels (u = somme, v = diff) pour construire le point S(u,v) en 2D.
    Les points sont triés de façon circulaire et aplatis pour former un descripteur.
    """
    uv_points = []
    for pt in intersections:
        idx = find_nearest_vertex(vertices, pt)
        uv_points.append([sums[idx], diffs[idx]])
    uv_points = np.array(uv_points)
    if len(uv_points) == 0:
        return None
    centroid = np.mean(uv_points, axis=0)
    angles = np.arctan2(uv_points[:,1]-centroid[1], uv_points[:,0]-centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_uv = uv_points[sorted_indices]
    descriptor = sorted_uv.flatten()
    return descriptor

###########################################
# Modes de l'application : Batch vs. Étude d'équivariance
###########################################

st.sidebar.header("Mode de l'application")
mode = st.sidebar.radio("Choisissez le mode", ("Batch Identification/Verification", "Étude de l'équivariance"))

###########################################
# Mode 1 : Batch Identification/Verification
###########################################
if mode == "Batch Identification/Verification":
    st.title("Batch de calcul des descripteurs S(u,v) sur BU3D-FE")
    st.markdown("""
    Ce mode permet de calculer en batch les descripteurs S(u,v) pour l'ensemble de la base BU3D‑FE (par exemple, 2500 maillages).
    Vous pouvez uploader plusieurs fichiers maillage et leurs repères associés.  
    Les descripteurs sont ensuite sauvegardés (au format npy) pour être utilisés dans des expériences d'identification et de vérification (Neutral vs. All).
    """)
    uploaded_meshes = st.file_uploader("Uploader les maillages (OBJ)", type=["obj"], accept_multiple_files=True)
    uploaded_bnds = st.file_uploader("Uploader les fichiers de repères (.bnd)", type=["bnd"], accept_multiple_files=True)
    # Paramètres communs (sensibilité 0.01)
    st.sidebar.header("Paramètres batch")
    target_sum = st.sidebar.number_input("Valeur cible pour 'somme'", value=100.0, step=0.01)
    tol_sum    = st.sidebar.slider("Tolérance pour 'somme'", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    target_diff = st.sidebar.number_input("Valeur cible pour 'diff'", value=20.0, step=0.01)
    tol_diff    = st.sidebar.slider("Tolérance pour 'diff'", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    n_samples = st.sidebar.number_input("Nombre de points pour B-spline", value=50, step=1)
    spline_s = st.sidebar.number_input("Paramètre de lissage B-spline", value=0.0, step=0.01)
    tol_inter = st.sidebar.slider("Tolérance pour l'intersection", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    
    if st.button("Lancer le batch"):
        if not uploaded_meshes or not uploaded_bnds:
            st.error("Veuillez uploader tous les fichiers nécessaires.")
        else:
            descriptors = {}
            for mesh_file in uploaded_meshes:
                # Trouver le fichier de repères correspondant (on suppose que les noms de fichiers partagent un identifiant commun)
                base_name = os.path.splitext(mesh_file.name)[0]
                # Recherche dans les uploaded_bnds du fichier dont le nom contient base_name
                matching_bnd = None
                for bnd_file in uploaded_bnds:
                    if base_name in bnd_file.name:
                        matching_bnd = bnd_file
                        break
                if matching_bnd is None:
                    st.warning(f"Fichier de repères non trouvé pour {mesh_file.name}.")
                    continue
                # Chargement du maillage et du fichier de repères
                with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp:
                    tmp.write(mesh_file.getvalue())
                    mesh_path = tmp.name
                mesh = load_mesh(mesh_path)
                vertices = np.array(mesh.vertices)
                faces = np.array(mesh.faces)
                landmarks = read_bnd_file(matching_bnd.getvalue())
                if landmarks is None:
                    st.warning(f"Erreur lecture repères pour {mesh_file.name}.")
                    continue
                # Extraction des repères pour les yeux intérieurs (indices par défaut 0 et 8)
                ref_left_point = landmarks[0]
                ref_right_point = landmarks[8] if landmarks.shape[0] > 8 else landmarks[-1]
                idx_left = find_nearest_vertex(vertices, ref_left_point)
                idx_right = find_nearest_vertex(vertices, ref_right_point)
                # Calcul de la représentation bipolaire
                somme, diff = compute_bipolar_representation(vertices, faces, idx_left, idx_right)
                sums = somme.numpy()
                diffs = diff.numpy()
                # Extraction des contours
                pts_sum = mesh.vertices[np.where(np.abs(sums - target_sum) < tol_sum)[0]]
                pts_diff = mesh.vertices[np.where(np.abs(diffs - target_diff) < tol_diff)[0]]
                if len(pts_sum) < 4 or len(pts_diff) < 4:
                    st.warning(f"Pas assez de points pour {mesh_file.name}.")
                    continue
                # Réparamétrisation forcée (fermeture)
                curve_sum, tck_sum, u_sum = fit_bspline_curve(pts_sum, n_samples=n_samples, s=spline_s, contour_type="sum", force_closed=True)
                curve_diff, tck_diff, u_diff = fit_bspline_curve(pts_diff, n_samples=n_samples, s=spline_s, contour_type="diff", force_closed=True)
                # Extraction des intersections (idéalement 4)
                intersections = find_four_intersections(curve_sum, curve_diff, tol=tol_inter)
                if intersections.size == 0:
                    st.warning(f"Aucune intersection trouvée pour {mesh_file.name}.")
                    continue
                # Construction du descripteur S(u,v)
                descriptor = get_Suv_descriptor(intersections, vertices, sums, diffs)
                if descriptor is not None:
                    descriptors[base_name] = descriptor
            if descriptors:
                st.write("Descripteurs calculés pour", len(descriptors), "maillages.")
                # Sauvegarde en format npy (dictionnaire)
                buffer = BytesIO()
                np.save(buffer, descriptors)
                buffer.seek(0)
                st.download_button("Télécharger les descripteurs (.npy)",
                                   data=buffer,
                                   file_name="descripteurs_Suv.npy",
                                   mime="application/octet-stream")
            else:
                st.write("Aucun descripteur n'a pu être généré.")

###########################################
# Mode 2 : Étude de l'équivariance/invariance
###########################################
if mode == "Étude de l'équivariance":
    st.title("Étude de l'équivariance de la description S(u,v)")
    st.markdown("""
    Dans ce mode, vous pouvez uploader un maillage et son fichier de repères, calculer le descripteur S(u,v),
    puis appliquer une transformation (par exemple, une rotation) et comparer le descripteur obtenu.
    
    **Théorie :**  
    Soit g ∈ SE(3) une transformation rigide et D(S) le descripteur S(u,v) de la face S.  
    L'équivariance signifie que D(g·S) = φ(g)·D(S), où φ(g) est l'action extérieure induite par g sur l'espace des descripteurs.
    Nous pouvons vérifier l'invariance (ou équivariance faible) en comparant les similarités (par exemple, le cosinus de l'angle) entre les descripteurs avant et après transformation.
    """)
    
    uploaded_mesh_eq = st.file_uploader("Maillage pour étude d'équivariance (OBJ)", type=["obj"], key="eq_mesh")
    uploaded_bnd_eq = st.file_uploader("Fichier de repères (.bnd)", type=["bnd"], key="eq_bnd")
    angle_deg = st.sidebar.slider("Angle de rotation (degrés)", min_value=0.0, max_value=360.0, value=30.0, step=0.01)
    
    # Paramètres d'extraction (sensibilité 0.01)
    target_sum_eq = st.sidebar.number_input("Valeur cible pour 'somme'", value=100.0, step=0.01, key="target_sum_eq")
    tol_sum_eq    = st.sidebar.slider("Tolérance pour 'somme'", min_value=0.01, max_value=10.0, value=1.0, step=0.01, key="tol_sum_eq")
    target_diff_eq = st.sidebar.number_input("Valeur cible pour 'diff'", value=20.0, step=0.01, key="target_diff_eq")
    tol_diff_eq    = st.sidebar.slider("Tolérance pour 'diff'", min_value=0.01, max_value=10.0, value=1.0, step=0.01, key="tol_diff_eq")
    n_samples_eq = st.sidebar.number_input("Nombre de points pour B-spline", value=50, step=1, key="n_samples_eq")
    spline_s_eq = st.sidebar.number_input("Paramètre de lissage B-spline", value=0.0, step=0.01, key="spline_s_eq")
    tol_inter_eq = st.sidebar.slider("Tolérance pour l'intersection", min_value=0.01, max_value=10.0, value=1.0, step=0.01, key="tol_inter_eq")
    
    if st.button("Lancer l'étude d'équivariance"):
        if uploaded_mesh_eq is None or uploaded_bnd_eq is None:
            st.error("Veuillez uploader le maillage et le fichier de repères.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp:
                tmp.write(uploaded_mesh_eq.getvalue())
                mesh_path_eq = tmp.name
            mesh_eq = load_mesh(mesh_path_eq)
            vertices_eq = np.array(mesh_eq.vertices)
            faces_eq = np.array(mesh_eq.faces)
            
            landmarks_eq = read_bnd_file(uploaded_bnd_eq.getvalue())
            if landmarks_eq is None:
                st.error("Erreur lors de la lecture des repères.")
            else:
                st.write("Premières lignes des repères :", landmarks_eq[:5])
                ref_left_pt = landmarks_eq[0]
                ref_right_pt = landmarks_eq[8] if landmarks_eq.shape[0]>8 else landmarks_eq[-1]
                idx_left_eq = find_nearest_vertex(vertices_eq, ref_left_pt)
                idx_right_eq = find_nearest_vertex(vertices_eq, ref_right_pt)
                st.write(f"Indices trouvés : œil gauche = {idx_left_eq}, œil droit = {idx_right_eq}")
                
                # Calcul du descripteur S(u,v) pour le maillage original
                somme_eq, diff_eq = compute_bipolar_representation(vertices_eq, faces_eq, idx_left_eq, idx_right_eq)
                sums_eq = somme_eq.numpy()
                diffs_eq = diff_eq.numpy()
                pts_sum_eq = mesh_eq.vertices[np.where(np.abs(sums_eq - target_sum_eq) < tol_sum_eq)[0]]
                pts_diff_eq = mesh_eq.vertices[np.where(np.abs(diffs_eq - target_diff_eq) < tol_diff_eq)[0]]
                if len(pts_sum_eq) < 4 or len(pts_diff_eq) < 4:
                    st.error("Pas assez de points pour le maillage original.")
                else:
                    curve_sum_eq, tck_sum_eq, u_sum_eq = fit_bspline_curve(pts_sum_eq, n_samples=n_samples_eq, s=spline_s_eq, contour_type="sum", force_closed=True)
                    curve_diff_eq, tck_diff_eq, u_diff_eq = fit_bspline_curve(pts_diff_eq, n_samples=n_samples_eq, s=spline_s_eq, contour_type="diff", force_closed=True)
                    intersections_eq = find_four_intersections(curve_sum_eq, curve_diff_eq, tol=tol_inter_eq)
                    descriptor_orig = get_Suv_descriptor(intersections_eq, vertices_eq, sums_eq, diffs_eq)
                    st.write("Descripteur original S(u,v) :", descriptor_orig)
                    
                    # Appliquer une transformation (rotation) au maillage
                    angle_rad = np.deg2rad(angle_deg)
                    # Rotation autour de l'axe Z par exemple
                    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                  [np.sin(angle_rad), np.cos(angle_rad), 0],
                                  [0, 0, 1]])
                    vertices_trans = (R @ vertices_eq.T).T
                    mesh_trans = mesh_eq.copy()
                    mesh_trans.vertices = vertices_trans
                    # Recalculez la représentation bipolaire sur le maillage transformé
                    somme_trans, diff_trans = compute_bipolar_representation(vertices_trans, faces_eq, idx_left_eq, idx_right_eq)
                    sums_trans = somme_trans.numpy()
                    diffs_trans = diff_trans.numpy()
                    pts_sum_trans = mesh_trans.vertices[np.where(np.abs(sums_trans - target_sum_eq) < tol_sum_eq)[0]]
                    pts_diff_trans = mesh_trans.vertices[np.where(np.abs(diffs_trans - target_diff_eq) < tol_diff_eq)[0]]
                    if len(pts_sum_trans) < 4 or len(pts_diff_trans) < 4:
                        st.error("Pas assez de points pour le maillage transformé.")
                    else:
                        curve_sum_trans, tck_sum_trans, u_sum_trans = fit_bspline_curve(pts_sum_trans, n_samples=n_samples_eq, s=spline_s_eq, contour_type="sum", force_closed=True)
                        curve_diff_trans, tck_diff_trans, u_diff_trans = fit_bspline_curve(pts_diff_trans, n_samples=n_samples_eq, s=spline_s_eq, contour_type="diff", force_closed=True)
                        intersections_trans = find_four_intersections(curve_sum_trans, curve_diff_trans, tol=tol_inter_eq)
                        descriptor_trans = get_Suv_descriptor(intersections_trans, vertices_trans, sums_trans, diffs_trans)
                        st.write("Descripteur après transformation S(u,v) :", descriptor_trans)
                        
                        # Étude de similarité : calcul du cosinus de l'angle entre les deux descripteurs
                        if descriptor_orig is not None and descriptor_trans is not None:
                            # Normalisation des descripteurs
                            desc_orig_norm = descriptor_orig / (norm(descriptor_orig)+1e-8)
                            desc_trans_norm = descriptor_trans / (norm(descriptor_trans)+1e-8)
                            similarity = np.dot(desc_orig_norm, desc_trans_norm)
                            st.write(f"Similarité (cosinus) entre les descripteurs : {similarity:.4f}")
                            st.markdown("""
                            **Analyse théorique :**  
                            Soit D(S) le descripteur S(u,v) d'une face S, et g ∈ SE(3) une transformation rigide.  
                            L'équivariance implique que D(g·S) = φ(g)·D(S), pour une certaine représentation linéaire φ(g).  
                            Ici, nous vérifions que la similarité entre D(S) et D(g·S) est élevée, indiquant que la transformation n'affecte pas significativement le descripteur.
                            """)
                        else:
                            st.write("Impossible de comparer les descripteurs.")
