import sys
import os  
env = os.environ    
current_path = os.getcwd()  
robouniview_path =  current_path  
env['PATH'] = env['PATH'] + ':'+  robouniview_path
sys.path.append(robouniview_path)

#from robouniview.new_data.data import CalvinDataset_pcd
import cv2
import numpy as np
import skimage.transform

from pathlib import Path

class OccupancyVFE:
    def __init__(self, voxel_range, voxel_size):
        """
        Args:
        voxel_range = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        voxel_size = [x_step, y_step, z_step]
        """
        self.voxel_range = np.array(voxel_range)
        self.voxel_size = np.array(voxel_size)
        self.image_range = np.ceil(
            (self.voxel_range[:, 1] - self.voxel_range[:, 0]) / voxel_size
        ).astype(np.int32)
       
    def generate(self, points,rgb):
   
        points_loc = points[:, :3]
        mask = np.logical_and(points_loc > self.voxel_range[:, 0], points_loc < self.voxel_range[:, 1])
        mask = mask.all(axis=-1)
        valid_points = points[mask]
        rgb = rgb[mask]
        valid_loc = valid_points[:, :3]
        coors = np.floor((valid_loc - self.voxel_range[:, 0]) / self.voxel_size)
        coors = np.clip(coors, 0, self.image_range - 1).astype(np.int32)
        occ_label = np.zeros([*self.image_range], dtype=np.float32)
        occ_label[coors[:, 0], coors[:, 1], coors[:, 2]] = 1
        r_label = np.zeros([*self.image_range], dtype=np.float32)
        r_label[coors[:, 0], coors[:, 1], coors[:, 2]] = rgb[:,0]
        g_label = np.zeros([*self.image_range], dtype=np.float32)
        g_label[coors[:, 0], coors[:, 1], coors[:, 2]] = rgb[:,1]
        b_label = np.zeros([*self.image_range], dtype=np.float32)
        b_label[coors[:, 0], coors[:, 1], coors[:, 2]] = rgb[:,2]
        grid_labels = np.stack([occ_label,r_label,g_label,b_label], -1)

        return grid_labels

    @staticmethod
    def decode_occupied_grid(label):
        grid = label[:,:,:,0]
        occupied_loc = np.where(grid > 0.5)
        occupied_points = np.stack(occupied_loc).T
        occupied_rgb = label[occupied_points[:, 0], occupied_points[:, 1], occupied_points[:, 2]][:,1:]

        return occupied_points,occupied_rgb
    
    def decode_occupied_grid_with_range(self, label):
        grid = label[:,:,:,0]
        grid_x, grid_y, grid_z = grid.shape
        occupied_loc = np.where(grid > 0.5)
        occupied_points = np.stack(occupied_loc).T
        occupied_rgb = label[occupied_points[:, 0], occupied_points[:, 1], occupied_points[:, 2]][:,1:]

        occupied_points = occupied_points.astype(np.float32)
        occupied_points[:, 0] = occupied_points[:, 0]/(grid_x-1)*(self.voxel_range[0, 1] - self.voxel_range[0, 0])+self.voxel_range[0, 0]
        occupied_points[:, 1] = occupied_points[:, 1]/(grid_y-1)*(self.voxel_range[1, 1] - self.voxel_range[1, 0])+self.voxel_range[1, 0]
        occupied_points[:, 2] = occupied_points[:, 2]/(grid_z-1)*(self.voxel_range[2, 1] - self.voxel_range[2, 0])+self.voxel_range[2, 0]

        return occupied_points,occupied_rgb
    

if __name__ == "__main__":
    import open3d as o3d
