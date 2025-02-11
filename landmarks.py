import numpy as np

def read_bnd(filepath):
    """
    Lire un fichier .bnd contenant les repères (landmarks) du maillage.
    
    On suppose ici que le fichier est binaire et stocke des coordonnées float32.
    Chaque point est représenté par 3 float32 (x, y, z).
    Adaptez ce code selon le format exact du fichier BU-3DFE.
    
    Args:
        filepath (str): chemin du fichier .bnd.
    
    Renvoie:
        landmarks (np.ndarray): tableau de forme (N, 3).
    """
    try:
        data = np.fromfile(filepath, dtype=np.float32)
        landmarks = data.reshape(-1, 3)
        return landmarks
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier .bnd: {e}")
        return None

def visualize_landmarks_on_mesh(vertices, landmarks):
    """
    Visualiser les landmarks sur le nuage de points d'un maillage en utilisant Plotly.
    
    Args:
        vertices (np.ndarray): tableau des sommets du maillage, shape (N, 3).
        landmarks (np.ndarray): tableau des repères, shape (M, 3).
    """
    import plotly.graph_objs as go
    
    fig = go.Figure()
    # Afficher les sommets du maillage
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(size=1, color='gray'),
        name='Maillage'
    ))
    # Afficher les landmarks
    fig.add_trace(go.Scatter3d(
        x=landmarks[:, 0],
        y=landmarks[:, 1],
        z=landmarks[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Landmarks'
    ))
    fig.update_layout(title="Visualisation des Landmarks sur le Maillage")
    fig.show()
