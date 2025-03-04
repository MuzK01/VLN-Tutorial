import os
import sys
import json
from typing import Tuple

import cv2
import numpy as np
import networkx as nx
import MatterSim

matterport_build_path = f"/projects/VLN-Tutorial/Matterport3DSimulator/build"  # Path to simulator

if matterport_build_path not in sys.path:
    sys.path.append(matterport_build_path)

scan_dir = '/projects/VLN-Tutorial/duet/datasets/Matterport3D/v1_unzip_scans'
connectivity_dir = '/projects/VLN-Tutorial/duet/datasets/R2R/connectivity'

def visualize_panorama_img(scan, viewpoint, heading, elevation):
    WIDTH = 80
    HEIGHT = 480
    pano_img = np.zeros((HEIGHT, WIDTH * 36, 3), np.uint8)
    VFOV = np.radians(55)
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.initialize()
    for n_angle, angle in enumerate(range(-175, 180, 10)):
        sim.newEpisode([scan], [viewpoint], [heading + np.radians(angle)], [elevation])
        state = sim.getState()
        im = state[0].rgb
        im = np.array(im)
        pano_img[:, WIDTH * n_angle : WIDTH * (n_angle + 1), :] = im #image in BGR format
    return pano_img

def get_panorama(scan, viewpoint, viewpoint_heading):
    # Get panorama image
    images = []
    for viewpoint_elevation in (np.pi / 2 * x for x in range(-1, 2)):
        im = visualize_panorama_img(scan, viewpoint, viewpoint_heading, viewpoint_elevation)
        images.append(im)

    return images

def orient_to_coord(
    heading: float,
    elevation: float,
    agent_heading: float,
    agent_elevation: float,
    img_height: int,
    img_width: int
    ) -> Tuple[int, int]:
    heading -= agent_heading
    elevation -= agent_elevation

    while heading > np.pi:
        heading -= 2 * np.pi
    while heading < -np.pi:
        heading += 2 * np.pi

    while elevation > np.pi:
        heading -= 2 * np.pi
    while elevation < -np.pi:
        elevation += 2 * np.pi

    
    first_coord = (heading / (2 * np.pi) + 0.5) * img_width  # img.shape[1]
    if first_coord < 0:
        first_coord += img_width
    second_coord = (0.5 - elevation / (np.pi / 1.1)) * img_height  # img.shape[0]
    return int(first_coord), int(second_coord)

def load_nav_graph(graph_path):
    with open(graph_path) as f:
        G = nx.Graph()
        positions = {}
        heights = {}
        data = json.load(f)
        for i, item in enumerate(data):
            if item["included"]:
                for j, conn in enumerate(item["unobstructed"]):
                    if conn and data[j]["included"]:
                        positions[item["image_id"]] = np.array([item["pose"][3], item["pose"][7], item["pose"][11]])
                        heights[item["image_id"]] = float(item["height"])
                        assert data[j]["unobstructed"][i], "Graph should be undirected"
                        G.add_edge(item["image_id"], data[j]["image_id"])
        nx.set_node_attributes(G, values=positions, name="position")
        nx.set_node_attributes(G, values=heights, name="height")
    return G

def get_neighbour_viewpoints_coords(scan, viewpointId):
    # /home/mrearle/repos/VLN-HAMT/datasets/R2R/connectivity/XcA2TqTSSAj_connectivity.json
    graph_path = f"{connectivity_dir}/{scan}_connectivity.json"
    graph: nx.Graph = load_nav_graph(graph_path)

    curr_node = graph.nodes[viewpointId] 
    # TODO: Obtener coord de curr node. Usar esto para arreglar posicionamiento de viewpoints.

    viewpoints = {}
    for reachable in graph.neighbors(viewpointId):
        node = graph.nodes[reachable]
        x = node["position"][0] - curr_node["position"][0]
        y = node["position"][1] - curr_node["position"][1]
        z = node["position"][2] - curr_node["position"][2]
        dist = np.sqrt(x ** 2 + y ** 2)

        heading = (np.pi / 2) - np.arctan2(y, x)
        heading -= (2 * np.pi) * np.floor((heading + np.pi) / (2 * np.pi))
        elevation = np.arctan2(z - node["height"], dist)
        viewpoints[reachable] = (heading, elevation)

    return viewpoints

def visualize_nav_graph(scan, save_dir):
    """
    Visualizes the navigation graph for a given scan in 2D and 3D.
    Args:
        scan (str): The scan ID (e.g., "1LXtFkjw3qL")
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    graph_path = f"{connectivity_dir}/{scan}_connectivity.json"
    G = load_nav_graph(graph_path)
    
    # Extract positions for all nodes
    pos_3d = nx.get_node_attributes(G, 'position')
    pos_2d = {node: (pos[0], pos[1]) for node, pos in pos_3d.items()}
    
    # Create 2D visualization
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos_2d, node_size=20, node_color='blue', with_labels=False)
    plt.title(f'2D Navigation Graph for {scan}')
    plt.savefig(f'{save_dir}/nav_graph_2d_{scan}.png')
    
    # Create 3D visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw edges
    for edge in G.edges():
        x = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
        y = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
        z = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
        ax.plot(x, y, z, 'b-', alpha=0.5)
    
    # Draw nodes
    x = [pos[0] for pos in pos_3d.values()]
    y = [pos[1] for pos in pos_3d.values()]
    z = [pos[2] for pos in pos_3d.values()]
    ax.scatter(x, y, z, c='red', s=20)
    
    ax.set_title(f'3D Navigation Graph for {scan}')
    plt.savefig(f'{save_dir}/nav_graph_3d_{scan}.png')
    plt.close('all')
    print(f"Navigation graph saved to {save_dir}/nav_graph_3d_{scan}.png")

if __name__ == "__main__":
    scan = "1LXtFkjw3qL"
    viewpointId = "0b22fa63d0f54a529c525afbf2e8bb25"
    viewpoint_heading = 0
    images = get_panorama(scan, viewpointId, viewpoint_heading)
    save_dir = "tmp"
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(f"{save_dir}/{scan}_{viewpointId}_{i}.png", img)
    print(f"Panorama saved to {save_dir}")

    # visualization of the nav graph
    visualize_nav_graph(scan, save_dir)