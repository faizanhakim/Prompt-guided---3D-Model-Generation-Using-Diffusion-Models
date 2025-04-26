import torch
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import cubify
import pandas as pd
import numpy as np
from tqdm import tqdm

class ShapeNetWithPrompts(torch.utils.data.Dataset):
    def __init__(self, shapenet_root, prompts_csv, synsets, split='train'):
       
        self.shapenet = ShapeNetCore(shapenet_root, version=2, synsets=synsets, load_textures=True)
        self.prompts_df = pd.read_csv(prompts_csv, index_col=0, dtype={'Model_Id': str, 'Description': str, 'Synset_Id': str})

        # Create mapping from model_id to prompt
        self.model_to_prompt = dict(zip(
            self.prompts_df.index,
            self.prompts_df['Description']
        ))
        
    def __len__(self):
        return len(self.shapenet)
    
    def __getitem__(self, idx):
        model = self.shapenet[idx]
        
        synset_id, model_id = self.shapenet[idx]["synset_id"], self.shapenet[idx]["model_id"]
        
        prompt = self.model_to_prompt.get(model_id, "")
        
        verts = [model['verts'].float()]
        faces = [model['faces'].long()]
        textures = model['textures']
        
        if 'textures' in model and model['textures'] is not None:
            if model['textures'].ndim == 2 and model['textures'].shape[1] == 3:
                textures = TexturesVertex(verts_features=[torch.from_numpy(model['textures']).float()])        
            elif model['textures'].ndim == 2 and model['textures'].shape[1] == 2:
                textures = TexturesVertex(verts_features=[torch.ones_like(verts[0]) * 0.5])  # Gray color
            else:
                textures = TexturesVertex(verts_features=[torch.ones_like(verts[0]) * 0.5])

        # Create PyTorch3D mesh
        mesh = Meshes(verts=verts, faces=faces, textures=textures)
        
        # Convert mesh to voxel grid (implement your own conversion)
        voxels = self.mesh_to_voxel(mesh)
        
        return voxels, prompt, model_id, synset_id
    
    def mesh_to_voxel(self, mesh, size=32, method='pointcloud', num_samples=100000, dilation_iter=1):
        
        # Normalize the mesh to fit in [-1, 1] cube
        verts = mesh.verts_packed()
        center = verts.mean(0)
        verts = verts - center
        scale = verts.abs().max()
        verts = verts / (scale * 1.1)  # Slightly smaller to ensure fit
        
        # Create normalized mesh
        norm_mesh = Meshes(
            verts=[verts],
            faces=[mesh.faces_packed()],
            textures=mesh.textures
        )
        
        if method == 'pointcloud':
            # Sample points from mesh surface
            points = sample_points_from_meshes(norm_mesh, num_samples=num_samples)[0]
            
            # Convert points to voxel indices
            voxel_grid = torch.zeros(size, size, size)
            
            # Map from [-1,1] to [0, size-1]
            indices = ((points + 1) * (size - 1) / 2).long()
            
            # Filter out-of-bounds indices
            valid = (indices >= 0) & (indices < size)
            valid = valid.all(dim=1)
            indices = indices[valid]
            
            # Mark occupied voxels
            voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            
            # Apply dilation to fill small holes
            if dilation_iter > 0:
                from scipy.ndimage import binary_dilation
                voxel_grid = torch.from_numpy(
                    binary_dilation(voxel_grid.numpy(), iterations=dilation_iter)
                ).float()
                
        elif method == 'cubify':
            # Use PyTorch3D's cubify operation
            voxel_grid = torch.zeros(size, size, size)
            
            # Convert mesh to voxels
            vox_mesh = cubify(norm_mesh, size, thresh=0.5)
            
            if len(vox_mesh) > 0:
                # Get vertices of voxelized mesh
                verts = vox_mesh.verts_packed()
                
                # Convert to voxel indices
                indices = ((verts + 1) * (size - 1) / 2).long()
                valid = (indices >= 0) & (indices < size)
                valid = valid.all(dim=1)
                indices = indices[valid]
                
                # Mark occupied voxels
                voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        else:
            raise ValueError(f"Unknown voxelization method: {method}")
        
        return voxel_grid

    def synset_id_to_label(self, id):
        "Convert synset Id to class label"
        return self.shapenet.synset_dict[id]
    
    def label_to_synset_id(self, label):

        "Convert class label to synset Id"
        return self.shapenet.synset_dict_inv[label]
    
    def render(self, idx, device, cameras, raster_settings, lights):
        """
        Render a mesh using PyTorch3D.
        """
        model_id = self.shapenet[idx]["model_id"]
        images_by_model_ids = self.shapenet.render(
        model_ids=[model_id],
        device=device,
        cameras=cameras,
        raster_settings=raster_settings,
        lights=lights,
        )

        return images_by_model_ids