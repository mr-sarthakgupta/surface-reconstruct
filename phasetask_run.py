import torch
from torch import nn
from skimage import measure
import open3d as o3d
import numpy as np
from math import sqrt
# import your libraries

from model import ImplicitNet

# instantiate the model and optimizer

def load_pointcloud(filename):
    pcd = o3d.io.read_point_cloud(filename)
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
    return points
    
def write_mesh(v,f,filename):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f))
    o3d.io.write_triangle_mesh(filename,mesh)
    
def write_pointcloud(p,filename):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    o3d.io.write_point_cloud(filename,pc)

# class ImplcitNetwork(nn.Module):
#     def __init__(self):
#         pass

#     def forward(self, x):
#         pass

#     def phase_loss(self, x):
#         pass

class FourierFeatureTransform(nn.Module):
    def __init__(self, input_dim, mapping_size, scale):
        super().__init__()
        self.B = torch.randn((input_dim, mapping_size)) * scale
    
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B.to(x.device)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PHASELoss(nn.Module):
    def __init__(self, epsilon=0.01, lambda_val=0.1, mu=0.1, ball_radius=0.025, iter_points = 10, use_normals=False, fourier_features = None):
        """
        Args:
            epsilon: Regularization parameter that controls smoothness
            lambda_val: Weight for the reconstruction loss
            mu: Weight for the normal/gradient constraint loss
            ball_radius: Radius of balls around point samples for reconstruction loss
            use_normals: If True, uses provided normals; otherwise enforces unit gradients
        """
        super(PHASELoss, self).__init__()
        self.epsilon = epsilon
        self.lambda_val = lambda_val
        self.mu = mu
        self.ball_radius = ball_radius
        self.use_normals = use_normals
        self.fourier_features = fourier_features
        
    def double_well_potential(self, x):
        return torch.sum(torch.pow(x, 2) - 2*torch.abs(x) + torch.ones_like(x))
    
    def reconstruction_loss(self, u, points, sample_count=400):
        """
        Args:
            u: Neural network representing the signed density
            points: Input point cloud (B x N x 3)
            sample_count: Number of points to sample in each ball
        """
        
        og_points = points.clone()

        random_offsets = torch.rand((sample_count, og_points.shape[-1]))
        random_offsets = random_offsets / torch.norm(random_offsets, dim=-1, keepdim=True)
        # random_offsets = random_offsets * self.ball_radius * torch.rand_like(random_offsets[..., :1])
        random_offsets = random_offsets * self.ball_radius 

        # Sample points within balls
        points = og_points.unsqueeze(1).expand(-1, sample_count, -1) + random_offsets.unsqueeze(0).expand(og_points.shape[0], -1, -1).cuda()
        
        u_values = []
        for point in points:
            u_values.append(u(point).abs().sum() / abs(self.ball_radius))
        return torch.stack(u_values, dim = 0).mean()

    def w(self, epsilon, u, x):
        u_outs = u(x.cuda())
        return -1 * sqrt(epsilon) * torch.log(torch.ones_like(u_outs).cuda() - torch.abs(u_outs)) * torch.sign(u_outs)

    def normal_loss(self, u, points, normals):
        points.requires_grad_(True)
        if self.use_normals is False:
            w_outs = self.w(self.epsilon, u, points)
            grad_outputs = torch.ones_like(w_outs)  
            w_grads = torch.autograd.grad(
                outputs=w_outs,
                inputs=points,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            grad_norm = torch.norm(w_grads, p = 2, dim = -1)
            return torch.mean(torch.abs(torch.ones_like(grad_norm) - grad_norm) ** 2)

        if self.use_normals:
            w_outs = self.w(self.epsilon, u, points)
            grad_outputs = torch.ones_like(w_outs)
            w_grads = torch.autograd.grad(
                outputs=w_outs,
                inputs=points,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            print(normals.shape, w_grads.shape)

            exit()

            return torch.mean(torch.norm(normals - w_grads, p = 2, dim = -1) ** 2)

    
    def gradient_loss(self, u, points, dim_range):
        """
        Args:
            u: Neural network representing the signed density
            w: Log-transformed function (-sqrt(epsilon) * log(1-|u|) * sign(u))
            range: Input point cloud
            normals: Surface normals (optional)
        """
        inte = torch.zeros((points.shape[0]))

        # Create 3D grid from the provided range
        x_min, x_max = dim_range[0], dim_range[1]
        y_min, y_max = dim_range[0], dim_range[1]
        z_min, z_max = dim_range[0], dim_range[1]

        steps = 10  # Number of steps in each dimension
        x_vals = torch.linspace(x_min, x_max, steps)
        y_vals = torch.linspace(y_min, y_max, steps)
        z_vals = torch.linspace(z_min, z_max, steps)

        # Volume element for integration
        dx = (x_max - x_min) / steps
        dy = (y_max - y_min) / steps
        dz = (z_max - z_min) / steps
        volume_element = dx * dy * dz

        # Create a grid of all points at once
        grid_points = torch.stack(torch.meshgrid([x_vals, y_vals, z_vals], indexing='ij')).permute(1, 2, 3, 0).reshape(-1, 3)
        grid_points.requires_grad_(True)

        # Process in batches to avoid memory issues
        batch_size = 1000
        total_points = grid_points.shape[0]
        inte_sum = torch.zeros(1).cuda()

        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            curr_batch = grid_points[batch_start:batch_end].cuda()
            
            # Forward pass
            u_outs = u(curr_batch)
            
            # Compute gradients
            grad_outputs = torch.ones_like(u_outs)
            gradients = torch.autograd.grad(
                outputs=u_outs,
                inputs=curr_batch,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Accumulate squared gradient norms
            inte_sum += torch.sum(torch.norm(gradients, dim=-1, p=2)**2)

        inte = inte_sum * volume_element

        return inte 
        
    def forward(self, model, points, dim_range, normals=None):
        """
        Computes the complete PHASE loss
        
        Args:
            model: Neural network model for signed density function
            points: Input point cloud
            normals: Surface normals (optional)
        """
        if self.fourier_features is not None:   
            u = lambda x: model(self.fourier_features(x))
        else:
            u = lambda x: model(x)
        
        double_well_term = self.double_well_potential(u(points))
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(u, points)
        
        # gradient loss
        gradient_loss = self.gradient_loss(u, points, dim_range)

        # normal loss
        normal_loss = self.normal_loss(u, points, normals)
                
        total_loss = self.epsilon * gradient_loss + double_well_term + self.lambda_val * recon_loss + self.mu * normal_loss
                
        return total_loss, {
            'grad_term': gradient_loss,
            'double_well': double_well_term,
            'reconstruction': recon_loss,
            'normal_constraint': normal_loss
        }



lam, eps, mu = [10, 0.01, 10]


def compute_chamfer_distance(pred_points, gt_points):
    """
    Args:
        pred_points (torch.Tensor): Predicted point cloud (N x 3)
        gt_points (torch.Tensor): Ground truth point cloud (M x 3)
    """
    # Ensure inputs are on the same device
    device = pred_points.device
    
    # Compute all pairwise distances
    pred_expanded = pred_points.unsqueeze(1)  # (N, 1, 3)
    gt_expanded = gt_points.unsqueeze(0)      # (1, M, 3)
    
    # Compute squared distances
    dist_matrix = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)  # (N, M)
    
    # Compute minimum distances in both directions
    dist_pred_to_gt = torch.min(dist_matrix, dim=1)[0]  # (N,)
    dist_gt_to_pred = torch.min(dist_matrix, dim=0)[0]  # (M,)
    
    # Average the distances (symmetric Chamfer distance)
    chamfer_dist = torch.mean(dist_pred_to_gt) + torch.mean(dist_gt_to_pred)
    
    return chamfer_dist.item()


def sdf_function(epsilon, u, x):
    u_outs = u(x.cuda())
    return -1 * sqrt(epsilon) * torch.log(torch.ones_like(u_outs).cuda() - torch.abs(u_outs)) * torch.sign(u_outs)

def sample_mesh_points(mesh_path, n_points=10000):
    """
    Args:
        mesh_path (str): Path to the mesh file
        n_points (int): Number of points to sample    
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
    return points

def evaluate_reconstruction(model, gt_mesh_path, resolution=64, bounds=(-2.0, 2.0), n_points=10000):
    """
    Args:
        model: Neural network model for implicit function
        gt_mesh_path (str): Path to ground truth mesh
        resolution (int): Resolution for marching cubes grid
        bounds (tuple): Min and max bounds for the grid
        n_points (int): Number of points to sample for Chamfer distance    
    """
    
    with torch.no_grad():
        # Create grid for marching cubes
        x = np.linspace(bounds[0], bounds[1], resolution)
        y = np.linspace(bounds[0], bounds[1], resolution)
        z = np.linspace(bounds[0], bounds[1], resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = torch.tensor(np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1), 
                              dtype=torch.float32)
        
        # Process in batches to avoid memory issues
        batch_size = 10000
        sdf_grid = []
        for i in range(0, points.shape[0], batch_size):
            batch_points = points[i:i+batch_size]
            sdf_batch = sdf_function(eps, model, batch_points).detach().cpu().numpy()
            sdf_grid.append(sdf_batch)
        
        sdf_grid = np.concatenate(sdf_grid, axis=0).reshape(resolution, resolution, resolution)
        
        # Generate mesh using marching cubes

        print("SDF minimum and maximum", sdf_grid.min(), sdf_grid.max())

        v, f, _, _ = measure.marching_cubes(sdf_grid, level = 0)
        
        # Scale vertices back to original coordinate system
        v = v / (resolution - 1) * (bounds[1] - bounds[0]) + bounds[0]
        
        # First save reconstructed mesh to temporary file
        temp_mesh_path = 'temp_reconstruction.ply'
        write_mesh(v, f, temp_mesh_path)
        
        # Sample points from both meshes
        pred_points = sample_mesh_points(temp_mesh_path, n_points)
        gt_points = sample_mesh_points(gt_mesh_path, n_points)
        
        # Compute Chamfer distance
        chamfer_dist = compute_chamfer_distance(pred_points, gt_points)
        
    return chamfer_dist, v, f

fourier = False
# fourier = True

if fourier:
    input_dim = 3
    mapping_size = 256
    scale = 10.0
    model = ImplicitNet(d_in = mapping_size * 2, dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ], skip_in = [4], geometric_init = True)
    fourier_features = FourierFeatureTransform(input_dim, mapping_size, scale)
    fourier_features.to("cuda:0")
else:
    model = ImplicitNet(d_in = 3, dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ], skip_in = [4], geometric_init = True)
    fourier_features = None


def analyze_model_weights(model):
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.extend(param.data.cpu().flatten().tolist())
    
    
    weights = torch.tensor(weights)
    stats = {
        'avg': weights.mean().item(),
        'median': weights.median().item(),
        'min': weights.min().item(),
        'max': weights.max().item(),
        'abs_avg': weights.abs().mean().item()
    }
    
    print(f"Model weights statistics:")
    print(f"  Average: {stats['avg']:.6f}")
    print(f"  Median:  {stats['median']:.6f}")
    print(f"  Min:     {stats['min']:.6f}")
    print(f"  Max:     {stats['max']:.6f}")
    print(f"  Abs Avg: {stats['abs_avg']:.6f}")
    
    return stats

analyze_model_weights(model)

opt = torch.optim.AdamW(
            [
                {
                    "params": model.parameters(),
                    "lr": 0.0005,
                    "weight_decay": 0
                },
            ])



iters=100000
loss_fn = PHASELoss(epsilon=eps, lambda_val=lam, mu=mu, ball_radius=0.0025, use_normals=True, fourier_features = fourier_features)

gt_mesh_path = "Preimage_Implicit_DLTaskData/meshes/armadillo.obj"

normals = True

if normals:
    # Load normals if available
    normals = load_pointcloud('Preimage_Implicit_DLTaskData/armadillo_normals_10000.xyz')
else:    
    normals = None

gt_points_all = sample_mesh_points(gt_mesh_path, n_points=10000)

model.train()

model.to("cuda:0")


points_range = (gt_points_all.min(), gt_points_all.max())

n_iter_points = 100

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100000, verbose=True)

for i in range(iters):
    idx = torch.randint(0, gt_points_all.shape[0], (n_iter_points,  ))
    selected_points = torch.gather(gt_points_all, 0, idx.unsqueeze(-1).expand(-1, 3)).to("cuda")

    # Zero gradients at the start of each iteration
    opt.zero_grad()
    
    # Forward pass and compute loss
    if normals is not None:
        loss, loss_components = loss_fn(model, selected_points, points_range, normals)
    else:
        loss, loss_components = loss_fn(model, selected_points, points_range)
    
    # Backward pass
    loss.backward()
   
    opt.step()
    scheduler.step()
    print(f"iter {i}", f"total_loss: {loss.item():.4f}", f"normal_loss: {loss_components['normal_constraint'].item():.4f}", f"gradient_loss: {loss_components['grad_term'].item():.4f}", f"double_well_term: {loss_components['double_well'].item():.4f}", f"recon_loss: {loss_components['reconstruction'].item():.4f}")
            
    if i %1000 == 0:
        # run evaluation 
        try:
            chamfer_dist, v, f = evaluate_reconstruction(model, gt_mesh_path, resolution=64, bounds=(-2.0, 2.0), n_points=10000)
            print(f"Chamfer distance: {chamfer_dist:.6f}")
            # create mesh with marching cubes
            write_mesh(v,f,f'intermediates/mesh_{i}_{chamfer_dist:.4f}.ply')
            # Save model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'iteration': i,
                'loss': loss.item(),
                'chamfer_dist': chamfer_dist,
            }, f'trained_models/model_checkpoint_{i}_{chamfer_dist}.pt')
            
            print(f"Iter {i}/{iters}, Loss: {loss.item():.6f}, "
                f"Grad: {loss_components['grad_term'].item():.6f}, "
                f"DW: {loss_components['double_well'].item():.6f}, "
                f"Recon: {loss_components['reconstruction'].item():.6f}, "
                f"Norm: {loss_components['normal_constraint'].item():.6f}")

        except:
            print("Error in evaluation")
        