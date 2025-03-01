import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PHASELoss(nn.Module):
    """
    Implementation of the PHASE loss for implicit neural representations
    from 'Phase Transitions, Distance Functions, and Implicit Neural Representations'
    by Yaron Lipman.
    
    This loss trains a neural network to learn a signed density function (u) that:
    1. Converges to a proper occupancy function in the limit
    2. Has a log transform (w) that converges to a signed distance function
    3. Has minimal surface perimeter
    4. Passes through the input point cloud data
    """
    def __init__(self, epsilon=0.01, lambda_val=0.1, mu=0.1, ball_radius=0.01, use_normals=False):
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
        
    def double_well_potential(self, u):
        return u**2 - 2*torch.abs(u) + 1
    
    def reconstruction_loss(self, u, points, sample_count=10):
        """
        Args:
            u: Neural network representing the signed density
            points: Input point cloud (B x N x 3)
            sample_count: Number of points to sample in each ball
        """
        batch_size = points.shape[0]
        n_points = points.shape[1]
        
        # Sample points from random subset of input points
        idx = torch.randint(0, n_points, (batch_size, sample_count))
        selected_points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        
        # Generate random offsets within ball_radius
        random_offsets = torch.randn_like(selected_points)
        random_offsets = random_offsets / torch.norm(random_offsets, dim=-1, keepdim=True)
        random_offsets = random_offsets * self.ball_radius * torch.rand_like(random_offsets[..., :1])
        
        # Sample points within balls
        sampled_points = selected_points + random_offsets
        
        # Evaluate network at sampled points
        u_values = u(sampled_points)
        
        # Compute average value in each ball
        return torch.mean(torch.abs(u_values))
    
    def gradient_loss(self, u, w, points, normals=None):
        """
        Computes the gradient constraint loss:
        - If normals are provided, aligns gradients with normals
        - Otherwise, enforces unit gradient norm
        
        Args:
            u: Neural network representing the signed density
            w: Log-transformed function (-sqrt(epsilon) * log(1-|u|) * sign(u))
            points: Input point cloud
            normals: Surface normals (optional)
        """
        points.requires_grad_(True)
        w_val = w(points)
        
        # Compute gradients of w with respect to input points
        grad_outputs = torch.ones_like(w_val)
        gradients = torch.autograd.grad(
            outputs=w_val,
            inputs=points,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        if self.use_normals and normals is not None:
            # Align gradients with provided normals
            return F.l1_loss(gradients, normals)
        else:
            # Enforce unit gradient norm
            gradient_norm = torch.norm(gradients, dim=-1)
            return torch.mean((gradient_norm - 1.0)**2)
    
    def forward(self, model, points, normals=None):
        """
        Computes the complete PHASE loss
        
        Args:
            model: Neural network model for signed density function
            points: Input point cloud
            normals: Surface normals (optional)
        """
        # Get the signed density function values
        u = lambda x: model(x)
        
        # Define the log-transformed function w (the smoothed SDF)
        w = lambda x: -torch.sqrt(self.epsilon) * torch.log(1 - torch.abs(u(x))) * torch.sign(u(x))
        
        # Sample random points in the domain for regularization term
        batch_size = points.shape[0]
        domain_points = torch.rand((batch_size, 1000, 3), device=points.device) * 2 - 1
        
        # Evaluate model on random domain points
        u_domain = u(domain_points)
        
        # Calculate the gradient of u
        domain_points.requires_grad_(True)
        grad_outputs = torch.ones_like(u_domain)
        grad_u = torch.autograd.grad(
            outputs=u_domain, 
            inputs=domain_points,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Double-well potential term
        double_well_term = torch.mean(self.double_well_potential(u_domain))
        
        # Gradient regularization term
        grad_term = torch.mean(self.epsilon * torch.sum(grad_u**2, dim=-1))
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(u, points)
        
        # Normal/gradient constraint loss
        normal_loss = self.gradient_loss(u, w, points, normals)
        
        # Total loss
        total_loss = grad_term + double_well_term + self.lambda_val * recon_loss + self.mu * normal_loss
        
        return total_loss, {
            'grad_term': grad_term.item(),
            'double_well': double_well_term.item(),
            'reconstruction': recon_loss.item(),
            'normal_constraint': normal_loss.item()
        }


# Example usage:
class ImplicitNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=512, num_layers=8):
        super(ImplicitNetwork, self).__init__()
        
        # First layer
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ])
        
        # Middle layers with skip connection at layer 4
        for i in range(1, num_layers-1):
            if i == 4:  # Skip connection
                self.layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            
        # Last layer
        self.layers.append(nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        """
        Forward pass that includes the skip connection
        """
        input_x = x
        
        # First layers until skip connection
        for i in range(8):  # 4 layers (linear + ReLU) = 8 operations
            x = self.layers[i](x)
            
        # Skip connection
        x = torch.cat([x, input_x], dim=-1)
        
        # Remaining layers
        for i in range(8, len(self.layers)):
            x = self.layers[i](x)
            
        return x.squeeze(-1)

# Training loop example
def train_implicit_surface(points, normals=None, epochs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and loss
    model = ImplicitNetwork().to(device)
    phase_loss = PHASELoss(
        epsilon=0.01,         # Controls smoothness
        lambda_val=0.1,       # Weight for reconstruction loss
        mu=0.1,               # Weight for normal constraint
        use_normals=normals is not None
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Convert inputs to torch tensors
    points = torch.tensor(points, dtype=torch.float32).to(device)
    if normals is not None:
        normals = torch.tensor(normals, dtype=torch.float32).to(device)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss, loss_terms = phase_loss(model, points, normals)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            for term, value in loss_terms.items():
                print(f"  {term}: {value:.6f}")
    
    return model

# Extract the surface as a mesh using Marching Cubes (requires additional libraries)
def extract_mesh(model, resolution=128, bbox_min=(-1,-1,-1), bbox_max=(1,1,1)):
    """
    Extract the surface as a mesh using marching cubes.
    Requires scikit-image or other implementation of marching cubes.
    """
    from skimage.measure import marching_cubes
    
    device = next(model.parameters()).device
    
    # Generate grid points
    x = torch.linspace(bbox_min[0], bbox_max[0], resolution).to(device)
    y = torch.linspace(bbox_min[1], bbox_max[1], resolution).to(device)
    z = torch.linspace(bbox_min[2], bbox_max[2], resolution).to(device)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    
    # Evaluate model
    with torch.no_grad():
        u_values = model(points)
        u_grid = u_values.reshape(resolution, resolution, resolution).cpu().numpy()
    
    # Convert to SDF-like values for marching cubes
    epsilon = 0.01  # Same as in the loss
    w_grid = -np.sqrt(epsilon) * np.log(1 - np.abs(u_grid)) * np.sign(u_grid)
    
    # Run marching cubes at 0 level-set
    vertices, faces, normals, _ = marching_cubes(w_grid, level=0)
    
    # Scale vertices back to original space
    scale = np.array(bbox_max) - np.array(bbox_min)
    vertices = vertices / (resolution - 1) * scale + np.array(bbox_min)
    
    return vertices, faces, normals