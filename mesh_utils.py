import open3d as o3d
import numpy as np

def load_mesh(filepath):
    """
    Charger un maillage à partir d'un fichier (OBJ, PLY, STL, etc.)
    """
    mesh = o3d.io.read_triangle_mesh(filepath)
    mesh.compute_vertex_normals()
    return mesh

def convert_o3d_to_plotly(mesh):
    """
    Convertir un maillage Open3D en données (vertices, triangles)
    utilisables pour Plotly.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    return vertices, triangles

def icp_registration(source_mesh, target_mesh, threshold=1.0, max_iteration=80):
    """
    Effectuer l'alignement ICP entre le maillage source et le maillage cible.
    Cette version convertit les TriangleMesh en PointCloud en échantillonnant des points
    uniformément avant d'exécuter ICP.

    Args:
        source_mesh (o3d.geometry.TriangleMesh): Maillage source.
        target_mesh (o3d.geometry.TriangleMesh): Maillage cible.
        threshold (float): Distance maximale pour une correspondance.
        max_iteration (int): Nombre maximum d'itérations ICP.

    Renvoie:
        transformation (np.ndarray): Matrice de transformation 4x4.
        eval_result (RegistrationResult): Résultat de l'ICP.
    """
    # Convertir les maillages en nuages de points en échantillonnant uniformément
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=30000)
    target_pcd = target_mesh.sample_points_uniformly(number_of_points=30000)

    # Vérifier les types
    assert isinstance(source_pcd, o3d.geometry.PointCloud), "source_pcd doit être un PointCloud"
    assert isinstance(target_pcd, o3d.geometry.PointCloud), "target_pcd doit être un PointCloud"

    # Initialisation de la transformation (identité)
    trans_init = np.eye(4, dtype=np.float64)

    # Exécuter ICP en utilisant la version point-to-point
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )

    return reg_p2p.transformation, reg_p2p


def apply_transformation(mesh, transformation):
    """
    Appliquer une transformation (matrice 4x4) à un maillage.
    """
    mesh.transform(transformation)
    return mesh
