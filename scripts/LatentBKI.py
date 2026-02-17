'''
refer to https://github.com/UMich-CURLY/LatentBKI/blob/main/Models/LatentBKI.py
'''

import torch
torch.backends.cudnn.deterministic = True

class LatentBKI(torch.nn.Module):
    def __init__(self, grid_size, min_bound, max_bound, filter_size=3,
                 num_classes=42, latent_dim=512, device="cpu", datatype=torch.float32,
                max_dist=0.5, kernel="sparse", pseduo_discrete = True):
        '''
        Input:
            grid_size: (x, y, z) int32 array, number of voxels
            min_bound: (x, y, z) float32 array, lower bound on local map
            max_bound: (x, y, z) float32 array, upper bound on local map
            filter_size: int, dimension of the kernel on each axis (must be odd)
            num_classes: int, number of classes
            prior: float32, value of prior in map
            device: cpu or gpu
            max_dist: size of the kernel ell parameter
            kernel: kernel to choose
            per_class: whether to learn a different kernel for each class
        '''
        super().__init__()
        self.min_bound = min_bound.view(-1, 3).to(device)
        self.max_bound = max_bound.view(-1, 3).to(device)
        self.grid_size = grid_size
        self.dtype = datatype

        self.kernel = kernel
        self.device = device
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.pseduo_discrete = pseduo_discrete
        
        self.voxel_sizes = (self.max_bound.view(-1) - self.min_bound.view(-1)) / self.grid_size.to(self.device)
        
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.ell = torch.tensor(max_dist, dtype=self.dtype, device=self.device, requires_grad=False)
        self.sigma = torch.tensor(1.0, device=self.device)  # Kernel must map to 0 to 1
        self.filter_size = torch.tensor(filter_size, dtype=torch.long, requires_grad=False, device=self.device)
                
        self.initialize_kernel()
        
        [xs, ys, zs] = [(max_bound[i]-min_bound[i])/(2*grid_size[i]) + 
                        torch.linspace(min_bound[i], max_bound[i], device=device, steps=grid_size[i]+1)[:-1] 
                        for i in range(3)]

        self.centroids = torch.cartesian_prod(xs, ys, zs).to(device)
    
    def initialize_kernel(self):
        # Initialize with sparse kernel
        assert(self.filter_size % 2 == 1)
        
        # Distances
        middle_ind = torch.floor(self.filter_size / 2)
        self.kernel_dists = torch.zeros([1, 1, self.filter_size, self.filter_size, self.filter_size],
                                            device=self.device)
        for x_ind in range(self.filter_size):
            for y_ind in range(self.filter_size):
                for z_ind in range(self.filter_size):
                    x_dist = torch.abs(x_ind - middle_ind) * self.voxel_sizes[0]
                    y_dist = torch.abs(y_ind - middle_ind) * self.voxel_sizes[1]
                    z_dist = torch.abs(z_ind - middle_ind) * self.voxel_sizes[2]
                    total_dist = torch.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
                    self.kernel_dists[0, 0, x_ind, y_ind, z_ind] = total_dist
        
    def sparse_kernel(self, d, ell, sigma):
        kernel_val = sigma * ((1.0/3)*(2 + torch.cos(2 * self.pi * d/ell))*(1 - d/ell) +
                              1.0/(2*self.pi) * torch.sin(2 * self.pi * d / ell))
        kernel_val[d >= ell] = 0
        return torch.clamp(kernel_val, min=0.0, max=1.0)
    
    def calculate_kernel_from_distance_vector(self, distance_vector):
        '''
        Input: 
            distance_vector: (n, 3)
            semantic_class: (n, 1)
        Output:
            kernel_val: (n, features_size)
        '''
        kernel_val = None
        ds = torch.norm(distance_vector, dim=-1, keepdim=True)
        if self.kernel == "sparse":
                kernel_val = self.sparse_kernel(ds, self.ell, self.sigma) 
        return kernel_val 

    def initialize_grid(self):
        mean_map = torch.zeros(self.grid_size[0], self.grid_size[1], self.grid_size[2], 
                        self.latent_dim, device=self.device, requires_grad=False,
                        dtype=self.dtype)
        variance_map = torch.ones(self.grid_size[0], self.grid_size[1], self.grid_size[2], 
                        self.latent_dim, device=self.device, requires_grad=False,
                        dtype=self.dtype)
        
        confidence_map = torch.zeros(self.grid_size[0], self.grid_size[1], self.grid_size[2], 
                        1, device=self.device, requires_grad=False,
                        dtype=self.dtype) 
        
        return (mean_map, variance_map, confidence_map)
    
    def grid_ind(self, input_pc, min_bound=None, max_bound=None):
        '''
        Input:
            input_xyz: N * (x, y, z, c) float32 array, point cloud
        Output:
            grid_inds: N' * (x, y, z, c) int32 array, point cloud mapped to voxels
        '''
        if min_bound is None:
            min_bound = self.min_bound
        if max_bound is None:
            max_bound = self.max_bound
        input_xyz   = input_pc[:, :3]
        labels      = input_pc[:, 3:]
        
        grid_inds = torch.floor((input_xyz - min_bound) / self.voxel_sizes)
        return torch.hstack((grid_inds, labels))
    
    def grid_to_continuous(self, grid_point, min_bound=None):
        '''
        Input:
            grid_inds: N' * (x, y, z, c) int32 array, point cloud mapped to voxels
        Output:
            grid_metric: N * (x, y, z, c) float32 array, grid point in metric space (same as point cloud)
        '''
        if min_bound == None:
            min_bound = self.min_bound
            
        grid_inds   = grid_point[..., :3]
        labels      = grid_point[..., 3:]
        
        grid_xyz = grid_inds * self.voxel_sizes + min_bound # TODO: Check the math is correct
        
        return torch.cat((grid_xyz, labels),dim=-1)
    
    def calculate_kernel(self, i=0):
        kernel_val = None
        kernel_val = self.sparse_kernel(self.kernel_dists, self.ell, self.sigma)
        return kernel_val
    
    def construct_neighbor_grid(self, grid_point, store_coresponding_index=False):
        '''
        Input:
            gird_point: (N, 3+feautre) int32 tensor, store the indicies of the grids
        Output:
            neighbor_grid_point: (N, filter_size^3, 3+feature+index), first elements cosresponding to all nigherbors of the first elemnts in the grid_points that recive the feature update
        '''
        # construct offsets
        offsets_range = torch.tensor(range(int(-(self.filter_size-1)/2), int((self.filter_size+1)/2)), device=self.device)
        offsets = torch.cartesian_prod(offsets_range, offsets_range, offsets_range) # TODO: redundent memory
        
        # append the coresponding index at the end of each grid point, so after boradcasting, the neigbors will all contain the indices of the original grid point that they are centered on 
        if store_coresponding_index:
            indicies = torch.arange(0, grid_point.shape[0], dtype=grid_point.dtype, device=self.device).reshape(-1,1)
            grid_point = torch.hstack((grid_point, indicies))
        
        # make the offsets the same shape as grid_point    
        pad_offsets = torch.zeros(offsets.shape[0], grid_point.shape[-1], device=self.device)
        pad_offsets[:,:3] = offsets
        
        # construct neighbor grids
        neighbor_grid_point = grid_point.reshape(-1,1,grid_point.shape[-1])
        neighbor_grid_point = neighbor_grid_point + pad_offsets # (number of points, number of neighbors, grid_indcies & feature)
        
        return neighbor_grid_point
    
    def latent_map_update(self, current_map, point_cloud, max_bound=None, min_bound=None):
        # mean_map, mean_confidence_map, variance_map, variance_confidence_map = current_map[0], current_map[1], current_map[2], current_map[3]
        mean_map, variance_map, confidence_map = current_map[0], current_map[1], current_map[2]
        
        if min_bound is None:
            min_bound = self.min_bound
        if max_bound is None:
            max_bound = self.max_bound
            
        grid_pc = self.grid_ind(point_cloud, min_bound=min_bound, max_bound=max_bound)
            
        # construct neighbor grids
        neighbor_grid_pc = self.construct_neighbor_grid(grid_pc) # (N, f^3, 3+feature)
        
        # construct valid mask
        valid_input_mask = torch.all((neighbor_grid_pc[:,:,:3] < self.grid_size) & (neighbor_grid_pc[:,:,:3] >= torch.tensor([0,0,0], device=self.device)), axis=-1)
        
        # turn index into position in metric space
        neighbor_grid_metric = self.grid_to_continuous(neighbor_grid_pc, min_bound)
        
        # construct distance vector, the last dimension now contain a vector from a point to its neighbor grid centroid
        if self.pseduo_discrete:
            # NOTE: TEST substitute point_cloud with grid_pc
            neighbor_grid_metric[:,:,:3] = neighbor_grid_metric[:,:,:3] - self.grid_to_continuous(grid_pc, min_bound).unsqueeze(1)[:,:,:3]
        else:
            neighbor_grid_metric[:,:,:3] = neighbor_grid_metric[:,:,:3] - point_cloud.unsqueeze(1)[:,:,:3] # (n, f^3, xyz + feature)
        
        # select the valid grids
        neighbor_grid_pc = neighbor_grid_pc[valid_input_mask]
        valid_neighbor_grid_indices = [*neighbor_grid_pc[:,:3].T.to(torch.long)] # (3+feature, m) list contain indices in each coordinate
        neighbor_grid_metric = neighbor_grid_metric[valid_input_mask] # (m, xyz+feature)
        
        # calculate kernel value
        kernel_vals = self.calculate_kernel_from_distance_vector(neighbor_grid_metric[:,:3]) # (m, 1) 
        
        # compute sum of weight for each voxel in this frame
        k_bar_map = torch.zeros_like(confidence_map, dtype=self.dtype, device=self.device).index_put_(valid_neighbor_grid_indices[:3], kernel_vals, accumulate=True) 
        
        # compute contribution of observed sample in this frame
        y_bar_map = torch.zeros_like(mean_map, dtype=self.dtype, device=self.device).index_put_(valid_neighbor_grid_indices[:3], neighbor_grid_metric[:,3:] * kernel_vals, accumulate=True) / (k_bar_map + 1e-6) # TODO ways to remove this 1e-6
        
        # update variance map
        unique_grid_index = [*torch.unique(neighbor_grid_pc[:,:3], dim=0).T.to(torch.long)] # list (3, K), let number of updated grid be K
        delta_update_perpoint = neighbor_grid_metric[:,3:] - y_bar_map[valid_neighbor_grid_indices[:3]] # (m, 512)
        delta_mean = y_bar_map[unique_grid_index] - mean_map[unique_grid_index] # (K, C) y_bar - miu_0 for all the grids that have been updated, 
        
        # sum up the values that belongs to the same grid
        E_bar = delta_mean * delta_mean # (K, C), element-wise product
        S_bar_map = torch.zeros_like(mean_map, dtype=self.dtype, device=self.device).index_put_(valid_neighbor_grid_indices[:3], kernel_vals * delta_update_perpoint * delta_update_perpoint, accumulate=True)
        S_bar = S_bar_map[unique_grid_index] # (K, C)
        
        updated_lambda = confidence_map[unique_grid_index]
        updated_k = k_bar_map[unique_grid_index]
        scaling_factor = (updated_lambda * updated_k) / (updated_lambda + updated_k + 1e-6) # TODO: prevent division by zero
        variance_map[unique_grid_index] += S_bar + scaling_factor * E_bar
        
        # update mean map 
        mean_map = (confidence_map * mean_map + k_bar_map * y_bar_map) / (confidence_map + k_bar_map + 1e-6) # TODO: prevent division by zero
                
        # update confidence map
        confidence_map += k_bar_map
        
        return (mean_map, variance_map, confidence_map)
        
    def forward(self, current_map, point_cloud, iterative = False, min_bound = None, max_bound = None):
        if iterative:
            # trade speed for memory
            batch_size = 100000 # 64
            # batch_size = 1000 # 512
            start = 0
            end = start + batch_size
            N = point_cloud.shape[0]
            while end < N:
                batch_point_cloud = point_cloud[start:end]
                current_map = self.latent_map_update(current_map, batch_point_cloud, min_bound=min_bound, max_bound=max_bound)
                start = end 
                end = min(start+batch_size, N)
            # process last part
            batch_point_cloud = point_cloud[start:end]
            current_map = self.latent_map_update(current_map, batch_point_cloud, min_bound=min_bound, max_bound=max_bound)
            return current_map
        else:
            # take too much memory
            return self.latent_map_update(current_map, point_cloud, min_bound=min_bound, max_bound=max_bound)

            
                
        
        