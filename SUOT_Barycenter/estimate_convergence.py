import torch
import torch.nn.functional as F
from torch.linalg import inv
from torch.autograd import grad
from scipy.linalg import expm
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if GPU is available and set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def generate_centered_gaussians(num_samples, num_dimensions, num_gaussians):
    gaussians = []
    covariance_matrices = []  # List to store covariance matrices
    means = []  # List to store means
    
    for _ in range(num_gaussians):
        # Generate a random semi-positive definite covariance matrix
        eigenvalues = torch.rand(num_dimensions, device=device).mul(0.9).add(0.1)
        covariance_matrix = torch.diag(eigenvalues)
        
        # Store the covariance matrix in the list
        covariance_matrices.append(covariance_matrix)
        # Generate random samples from the multivariate normal distribution
        samples = torch.distributions.MultivariateNormal(torch.zeros(num_dimensions, device=device), covariance_matrix).sample((num_samples,))
        gaussians.append(samples.type(torch.complex64))

        # Calculate and store the mean of the samples
        mean = samples.mean(dim=0)
        means.append(mean.type(torch.complex64))
    
    return gaussians, covariance_matrices, means

def square_root(A):
    with torch.autograd.set_detect_anomaly(True):
        A = torch.tensor(A, device = device)
        eigenvalues, eigenvectors = torch.linalg.eig(A)

        # Compute the square root of the eigenvalues
        sqrt_eigenvalues = torch.sqrt(eigenvalues)

        # Construct the diagonal matrix with square root eigenvalues
        sqrt_diag = torch.diag(sqrt_eigenvalues)

        # Reconstruct matrix B using eigenvectors and square root eigenvalues
        B = eigenvectors @ sqrt_diag @ inv(eigenvectors)

    return B

def geometric_mean(sigma, sigma_prime):
    with torch.autograd.set_detect_anomaly(True):

        inv_sqrt_sigma = inv(square_root(sigma)).type(torch.complex64)
        sigma_prime = sigma_prime.type(torch.complex64)


        matrix_inside_sqrt = inv_sqrt_sigma @ sigma_prime @ inv_sqrt_sigma

        sqrt_matrix_inside_sqrt = square_root(matrix_inside_sqrt)

        geometric_mean_matrix = square_root(sigma) @ sqrt_matrix_inside_sqrt @ square_root(sigma)

    return geometric_mean_matrix

def UOT_matric(alpha_cov, beta_cov, tau):
    with torch.autograd.set_detect_anomaly(True):
        d = len(alpha_cov)
        Sigma_alpha_tau = (torch.eye(d, device=device) + (tau / 2) * inv(alpha_cov)).type(torch.complex64)
        Sigma_alpha_tau_beta = inv(square_root(beta_cov) + 1e-6 * torch.eye(d, device=device)) @ Sigma_alpha_tau @ inv(square_root(beta_cov) + 1e-6 * torch.eye(d, device=device))
        Sigma_alpha_tau_beta_inv = inv(Sigma_alpha_tau_beta)

        inner_term = torch.eye(d, device=device) + square_root(torch.eye(d, device=device) + 2 * tau * Sigma_alpha_tau_beta)
        Sigma_gamma = (tau / 2) * torch.eye(d, device=device) + (1 / 2) * Sigma_alpha_tau_beta_inv @ inner_term

        Sigma_x = inv(square_root(beta_cov) + 1e-6 * torch.eye(d, device=device)) @ Sigma_alpha_tau_beta_inv @ Sigma_gamma @ inv(square_root(beta_cov) + 1e-6 * torch.eye(d, device=device))

    return Sigma_x

def UOT_distance_2(alpha_cov, beta_cov, tau):
    d = len(alpha_cov)
    Sigma_alpha_tau = (torch.eye(d, device=device) + (tau / 2) * inv(alpha_cov)).type(torch.complex64)
    Sigma_x = UOT_matric(alpha_cov, beta_cov, tau)

    cov_term = torch.trace(Sigma_x @ Sigma_alpha_tau)
    cov_term -= 2 * torch.trace(geometric_mean(Sigma_x, beta_cov))
    cov_term -= tau / 2 * torch.log(torch.det(Sigma_x))
    cov_term += torch.trace(beta_cov)
    cov_term += tau / 2 * torch.log(torch.det(alpha_cov))

    Upsilon = cov_term - tau * d / 2
    uot_distance = tau * (1 - torch.exp(-Upsilon / tau))

    return uot_distance

def UOT_distance_toset(covariance_matrices, beta_cov, tau):
    d = len(beta_cov)
    n = len(covariance_matrices)
    total_distance = 0
    for k in range(n):
        distance = UOT_distance_2(covariance_matrices[k], beta_cov, tau) 
        total_distance += distance
    return 1 / n * total_distance

def transport_gradient(A, B):
    d = len(A)
    return geometric_mean(A, B) - torch.eye(d, device=device)

def matrix_inner_product(A, B, C):
    return torch.trace(A.t() @ C @ B)

def constant_speed_geodesic(A, B, num_steps, epsilon):
    d = len(A)
    B = B.type(torch.complex64)
    A = A.type(torch.complex64)
    geodesic = []
    t_value = []
    for t in torch.linspace(0, 1 + epsilon, num_steps, device=device):
        t_value.append(t.item())
        point = ((1 - t) * torch.eye(d, device=device) + t * geometric_mean(inv(A + 1e-6 * torch.eye(d, device=device)), B)) @ A @ (
                    (1 - t) * torch.eye(d, device=device) + t * geometric_mean(inv(A + 1e-6 * torch.eye(d, device=device)), B))
        geodesic.append(point)
    return geodesic, t_value

# Gradient Functions

def gradient_trace(Sigma_beta):
    d = len(Sigma_beta)
    return 2 * torch.eye(d, device=device)

def gradient_trace_2(Sigma_alpha, Sigma_beta, tau):
    d = len(Sigma_alpha)
    Sigma_alpha_tau = (torch.eye(d, device=device) + (tau / 2) * inv(Sigma_alpha)).type(torch.complex64)
    Sigma_beta = Sigma_beta.type(torch.complex64)

    term = square_root(inv(Sigma_alpha_tau)) @ Sigma_beta @ square_root(inv(Sigma_alpha_tau))
    Sigma_beta_alpha_tau_square = term @ term + 2 * tau * term
    Sigma_beta_alpha_tau = square_root(Sigma_beta_alpha_tau_square)

    M = square_root(inv(Sigma_alpha_tau)) @ inv(Sigma_beta_alpha_tau) @ square_root(inv(Sigma_alpha_tau))
    U = inv(Sigma_alpha_tau) @ Sigma_beta @ M + M @ Sigma_beta @ inv(Sigma_alpha_tau)

    result1 = inv(Sigma_alpha_tau)
    result2 = U + tau * M
    result =  result1 + 1 / 2 * result2

    return result

def gradient_logdet(Sigma_alpha, Sigma_beta, tau):
    d = len(Sigma_alpha)
    Sigma_alpha_tau = (torch.eye(d, device=device) + (tau / 2) * inv(Sigma_alpha)).type(torch.complex64)
    Sigma_beta = Sigma_beta.type(torch.complex64)

    term = square_root(Sigma_alpha_tau) @ inv(Sigma_beta) @ square_root(Sigma_alpha_tau)
    V = square_root(torch.eye(d, device=device) + tau * term)

    P = inv(Sigma_beta) @ square_root(Sigma_alpha_tau) @ inv(torch.eye(d, device=device) + V) @ square_root(Sigma_alpha_tau) @ inv(Sigma_beta)
    Q = square_root(Sigma_alpha_tau) @ inv(torch.eye(d, device=device) + V) @ square_root(Sigma_alpha_tau) @ inv(Sigma_beta) @ inv(Sigma_beta)

    result1 = -6 * inv(Sigma_beta)
    result2 = -2 * tau * (P + Q)

    result = result1 + result2
    return result

def gaussian_gradient(Sigma_alpha, Sigma_beta, tau):
    return gradient_trace(Sigma_beta) - gradient_trace_2(Sigma_alpha, Sigma_beta, tau) - tau / 4 * gradient_logdet(
        Sigma_alpha, Sigma_beta, tau)

def gaussian_gradient_auto(Sigma_alpha, Sigma_beta, tau):
    Sigma_beta.requires_grad_(True)
    # Compute the UOT_distance_2
    uot_distance = UOT_distance_2(Sigma_alpha, Sigma_beta, tau).type(torch.float64)

    # Compute the gradients
    eu_gradients = grad(uot_distance, Sigma_beta)[0] # Gradients are stored in gradients[0]
    gradient = torch.mm(eu_gradients, Sigma_beta)
    bures_gradient = 2 * (gradient + gradient.t())
    bures_gradient = bures_gradient.type(torch.complex64)
    return bures_gradient


def gaussian_gradient_barycenter(covariance_matrices, beta_cov, tau):
    d = len(beta_cov)
    n = len(covariance_matrices)
    barycenter_gradient = 0
    for k in range(n):
        gradient = gaussian_gradient(covariance_matrices[k], beta_cov, tau)
        barycenter_gradient += gradient
    return 1 / n * barycenter_gradient

def gaussian_gradient_barycenter_auto(covariance_matrices, beta_cov, tau):
    d = len(beta_cov)
    n = len(covariance_matrices)
    barycenter_gradient = 0
    for k in range(n):
        gradient = gaussian_gradient_auto(covariance_matrices[k], beta_cov, tau)
        barycenter_gradient += gradient
    return 1 / n * barycenter_gradient

# Manifold Operator Functions

def matrix_exponential(N):
    eigenvalues, U = torch.linalg.eig(N)
    Sigma = torch.diag(torch.exp(eigenvalues))
    expm_N = U @ Sigma @ U.T
    return expm_N

def retraction(M, zeta):
    d = len(M)
    M_sqrt = square_root(M)
    M_inv_sqrt = inv(M_sqrt + 1e-6 * torch.eye(d, device=device))
    temp = M_inv_sqrt @ zeta @ M_inv_sqrt
    exp_temp = matrix_exponential(temp)
    result = M_sqrt @ exp_temp @ M_sqrt
    return result

def orthogonal_projection(M, nabla_M):
    M = M.type(torch.float64)
    nabla_M = nabla_M.type(torch.float64)
    projected_gradient = (M @ (0.5 * (nabla_M + nabla_M.t())) @ M).type(torch.complex64)
    return projected_gradient

def lyapunov_operator(X, U):
    n = X.shape[0]
    
    # Create the Kronecker product of X with itself
    I = torch.eye(n, device = device)
    S = torch.kron(I, X) + torch.kron(X, I)
    S = S.type(torch.complex64)
    # Vectorize U
    vec_U = U.flatten()
    vec_X_solution = torch.linalg.solve(S,vec_U.unsqueeze(1))

    # Reshape the solution
    X_solution = vec_X_solution.reshape(n, n)

    return X_solution

def exponential_map_manifold(X,U):
    X = X.type(torch.complex64)
    lyapunov = lyapunov_operator(X,U)
    #print(lyapunov)
    term = torch.mm(torch.mm(lyapunov, X),lyapunov)
    return X + U + term
    

### Finding Barycenter ###

###################### EXACT GEODESIC GRADIENT DESCENT ##########################

def Bures_manifold_UOT_bary_gradientmethod_exactgrad(covariance_matrices, Sigma_beta_0, tau=1e-2, T=100, eta=1, epsilon=1e-20, early_stop = False):
    Sigma_beta = Sigma_beta_0
    n = len(covariance_matrices)
    uot_record = [UOT_distance_toset(covariance_matrices, Sigma_beta, tau)]
    for i in tqdm(range(T), desc="Processing exactgrad"):
        if early_stop == True:
            if i > 2:
                if torch.norm(uot_record[-2] - uot_record[-1]) < epsilon:
                    return Sigma_beta, uot_record
        direction = - eta * orthogonal_projection(Sigma_beta, gaussian_gradient_barycenter(covariance_matrices, Sigma_beta, tau=1e-2))
        Sigma_beta = retraction(Sigma_beta, direction)
        uot_dis = UOT_distance_toset(covariance_matrices, Sigma_beta, tau)
        uot_record.append(uot_dis)
    #print(uot_record)
    return Sigma_beta, uot_record

def Bures_manifold_UOT_bary_gradientmethod_exactgrad_SGD(covariance_matrices, Sigma_beta_0, tau=1e-2, T=100, eta=1, epsilon=1e-7, early_stop = False):
    Sigma_beta = Sigma_beta_0
    n = len(covariance_matrices)
    uot_record = [UOT_distance_toset(covariance_matrices, Sigma_beta, tau)]
    for i in tqdm(range(T), desc="Processing exactgrad_SGD"):
        if early_stop == True:
            if i > 2:
                if torch.norm(uot_record[-2] - uot_record[-1]) < epsilon:
                    return Sigma_beta, uot_record
        direction = - eta * orthogonal_projection(Sigma_beta, gaussian_gradient(covariance_matrices[i%n], Sigma_beta, tau=1e-2))
        Sigma_beta = retraction(Sigma_beta, direction)
        uot_dis = UOT_distance_toset(covariance_matrices, Sigma_beta, tau)
        uot_record.append(uot_dis)
    #print(uot_record)
    return Sigma_beta, uot_record

def Bures_manifold_UOT_bary_gradientmethod_exactgrad_adas(covariance_matrices, Sigma_beta_0, tau=1e-2, T=100, eta=1, epsilon=1e-20, early_stop=False):
    Sigma_beta = Sigma_beta_0
    n = len(covariance_matrices)
    uot_record = [UOT_distance_toset(covariance_matrices, Sigma_beta, tau)]

    # Add tqdm progress bar
    with tqdm(range(T), desc="Processing UOT_bary_exactgrad_adas", unit="step") as pbar:
        for i in pbar:
            if early_stop:
                if i > 2:
                    if torch.norm(uot_record[-2] - uot_record[-1]) < epsilon:
                        pbar.set_description("Early stop triggered")
                        return Sigma_beta, uot_record
            
            # Compute the direction
            direction = - eta * orthogonal_projection(Sigma_beta, gaussian_gradient_barycenter(covariance_matrices, Sigma_beta, tau=1e-2))
            
            # Update Sigma_beta using retraction
            new_Sigma_beta = retraction(Sigma_beta, direction)
            
            # Compute the new UOT distance
            uot_dis = UOT_distance_toset(covariance_matrices, new_Sigma_beta, tau)
            
            # Check if UOT distance increased
            if uot_dis.real > uot_record[-1].real:
                # If increased, do not save it and reduce step size by half
                eta = eta * 0.8
                pbar.set_postfix_str(f"Step size halved: eta = {eta:.6f}")
            else:
                # If not increased, update Sigma_beta and save the new UOT distance
                Sigma_beta = new_Sigma_beta
                uot_record.append(uot_dis)
                pbar.set_postfix_str(f"SUOT distance: {uot_dis:.6f}")
    
    return Sigma_beta, uot_record

###################### AUTO GRADIENT DESCENT ##########################

def Bures_manifold_UOT_bary_gradientmethod_autograd(covariance_matrices, Sigma_beta_0, tau=1e-2, T=100, eta=1, epsilon=1e-20, early_stop = False): #Follow to https://openreview.net/pdf?id=ZCHxGFmc62a
    Sigma_beta = Sigma_beta_0 
    n = len(covariance_matrices)
    uot_record = [UOT_distance_toset(covariance_matrices, Sigma_beta, tau)]
    for i in tqdm(range(T), desc="Processing autograd"):
        if early_stop == True:
            if i > 2:
                if torch.norm(uot_record[-2] - uot_record[-1]) < epsilon:
                    return Sigma_beta, uot_record
        direction = -eta * gaussian_gradient_barycenter_auto(covariance_matrices, Sigma_beta, tau=1e-2)
        Sigma_beta = exponential_map_manifold(Sigma_beta, direction)
        uot_dis = UOT_distance_toset(covariance_matrices, Sigma_beta, tau)
        uot_record.append(uot_dis)
    #print(uot_record)
    return Sigma_beta, uot_record

def Bures_manifold_UOT_bary_gradientmethod_autograd_SGD(covariance_matrices, Sigma_beta_0, tau=1e-2, T=100, eta=1, epsilon=1e-20, early_stop = False):
    Sigma_beta = Sigma_beta_0 
    n = len(covariance_matrices)
    uot_record = [UOT_distance_toset(covariance_matrices, Sigma_beta, tau)]
    for i in tqdm(range(T), desc="Processing autograd_SGD"):
        if early_stop == True:
            if i > 2:
                if torch.norm(uot_record[-2] - uot_record[-1]) < epsilon:
                    return Sigma_beta, uot_record
        direction = -eta * gaussian_gradient_auto(covariance_matrices[i%n], Sigma_beta, tau=1e-2)
        Sigma_beta = exponential_map_manifold(Sigma_beta, direction)
        uot_dis = UOT_distance_toset(covariance_matrices, Sigma_beta, tau)
        uot_record.append(uot_dis)
    #print(uot_record)
    return Sigma_beta, uot_record

def Bures_manifold_UOT_bary_gradientmethod_autograd_momentum(covariance_matrices, Sigma_beta_0, tau=1e-2, T=100, eta=1, epsilon=1e-20, momen_rate = 0.55, early_stop = False): 
    Sigma_beta = Sigma_beta_0 
    n = len(covariance_matrices)
    uot_record = [UOT_distance_toset(covariance_matrices, Sigma_beta, tau)]
    direction_list = [gaussian_gradient_barycenter_auto(covariance_matrices, Sigma_beta, tau=1e-2)]
    for i in tqdm(range(T), desc="Processing autograd_momentum"):
        if early_stop == True:
            if i > 2:
                if torch.norm(uot_record[-2] - uot_record[-1]) < epsilon:
                    return Sigma_beta, uot_record
        direction = - (momen_rate* direction_list[-1] + eta * gaussian_gradient_barycenter_auto(covariance_matrices, Sigma_beta, tau=1e-2))
        Sigma_beta = exponential_map_manifold(Sigma_beta, direction)
        uot_dis = UOT_distance_toset(covariance_matrices, Sigma_beta, tau)
        uot_record.append(uot_dis)
        direction_list.append(direction)
    #print(uot_record)
    return Sigma_beta, uot_record

def Bures_manifold_UOT_bary_gradientmethod_autograd_momentum_SGD(covariance_matrices, Sigma_beta_0, tau=1e-2, T = 100, eta=1, epsilon=1e-20, momen_rate = 0.55, early_stop = False): 
    Sigma_beta = Sigma_beta_0 
    n = len(covariance_matrices)
    uot_record = [UOT_distance_toset(covariance_matrices, Sigma_beta, tau)]
    direction_list = [gaussian_gradient_barycenter_auto(covariance_matrices, Sigma_beta, tau=1e-2)]
    for i in tqdm(range(T), desc="Processing autograd_momentum_SGD"):
        if early_stop == True:
            if i > 2:
                if torch.norm(uot_record[-2] - uot_record[-1]) < epsilon:
                    return Sigma_beta, uot_record
        direction = - (momen_rate* direction_list[-1] + eta * gaussian_gradient_auto(covariance_matrices[i%n], Sigma_beta, tau=1e-2))
        Sigma_beta = exponential_map_manifold(Sigma_beta, direction)
        uot_dis = UOT_distance_toset(covariance_matrices, Sigma_beta, tau)
        uot_record.append(uot_dis)
        direction_list.append(direction)
    #print(uot_record)
    return Sigma_beta, uot_record

########################### HYBRID GRADIENT DESCENT #################################

def OT_distance(A, B):
    dis = torch.trace(A + B - 2 * A @ geometric_mean(inv(A), B))
    return dis

def OT_distance_toset(covariance_matrices, beta_cov):
    d = len(beta_cov)
    n = len(covariance_matrices)
    total_distance = 0
    for k in range(n):
        distance = OT_distance(covariance_matrices[k], beta_cov)
        total_distance += distance
    return 1 / n * total_distance

def Bures_manifold_OT_bary(covariance_matrices, Sigma_beta_0, T=100, eta = 1):
    Sigma = Sigma_beta_0.type(torch.complex64)
    ot_record = [OT_distance_toset(covariance_matrices, Sigma)]
    d = len(Sigma_beta_0)
    n = len(covariance_matrices)
    for i in range(T):
        S = torch.zeros((d, d), device = device).type(torch.complex64)
        for k in range(len(covariance_matrices)):
            S += geometric_mean(inv(Sigma), covariance_matrices[k])
        S = 1 / n * S
        S = (1-eta)* torch.eye(d, device = device) + eta * S
        Sigma = S @ Sigma @ S
        ot_dis = OT_distance_toset(covariance_matrices, Sigma)
        ot_record.append(ot_dis)
    return Sigma


def Bures_manifold_OT_bary_SGD(covariance_matrices, Sigma_beta_0, eta = 1):
    Sigma = Sigma_beta_0.type(torch.complex64)
    ot_record = [OT_distance_toset(covariance_matrices, Sigma)]
    d = len(Sigma_beta_0)
    n = len(covariance_matrices)
    for i in range(n):
        S = geometric_mean(inv(Sigma), covariance_matrices[i])
        Scale = (1-eta)* torch.eye(d, device = device) + eta * S
        Sigma = Scale @ Sigma @ Scale
        ot_dis = OT_distance_toset(covariance_matrices, Sigma)
        ot_record.append(ot_dis)
    return Sigma


def Bures_manifold_UOT_bary_hybridmethod(covariance_matrices, Sigma_beta_0, tau=1e-2, T=100, eta=0.1, epsilon=1e-7, early_stop = False):
    Sigma_beta = Sigma_beta_0
    loss_record = [UOT_distance_toset(covariance_matrices, Sigma_beta, tau)]
    for i in tqdm(range(T), desc="Processing hybridmethod"):
        if early_stop == True:
            if i > 2:
                if torch.norm(loss_record[-2] - loss_record[-1]) < epsilon:
                    return Sigma_beta, loss_record
        covariance_matrices_new = [UOT_matric(Sigma_alpha, Sigma_beta, tau) for Sigma_alpha in covariance_matrices]
        Sigma_beta = Bures_manifold_OT_bary(covariance_matrices_new, Sigma_beta, eta = eta)
        loss_dis = UOT_distance_toset(covariance_matrices, Sigma_beta, tau)
        loss_record.append(loss_dis)
    #print(loss_record)
    return Sigma_beta, loss_record

def Bures_manifold_UOT_bary_hybridmethod_SGD(covariance_matrices, Sigma_beta_0, tau=1e-2, T = 100, eta=0.1, epsilon=1e-40, early_stop = False):
    n = len(covariance_matrices) 
    Sigma_beta = Sigma_beta_0
    loss_record = [UOT_distance_toset(covariance_matrices, Sigma_beta, tau)]
    for i in tqdm(range(T), desc="Processing hybridmethod_SGD"):
        if early_stop == True:
            if i > 2:
                if torch.norm(loss_record[-2] - loss_record[-1]) < epsilon:
                    return Sigma_beta, loss_record
        covariance_matrices_new = [UOT_matric(Sigma_alpha, Sigma_beta, tau) for Sigma_alpha in covariance_matrices]
        Sigma_beta = Bures_manifold_OT_bary_SGD(covariance_matrices_new, Sigma_beta, eta = eta)
        loss_dis = UOT_distance_toset(covariance_matrices, Sigma_beta, tau)
        loss_record.append(loss_dis)
    #print(loss_record)
    return Sigma_beta, loss_record

def Bures_manifold_UOT_bary_mini_hybridmethod(covariance_matrices, Sigma_beta_0, tau=1e-2, T=1000, eta=0.1, epsilon=1e-20, early_stop = False): #Not fully OT-barycenter update at each iteration
    Sigma_beta = Sigma_beta_0.type(torch.complex64)
    loss_record = [UOT_distance_toset(covariance_matrices, Sigma_beta, tau)]
    d = len(Sigma_beta_0)
    n = len(covariance_matrices)
    for i in tqdm(range(T), desc="Processing mini_hybridmethod"):
        if early_stop == True:
            if i > 2:
                if torch.norm(loss_record[-2] - loss_record[-1]) < epsilon:
                    return Sigma_beta, loss_record
        covariance_matrices_new = [UOT_matric(Sigma_alpha, Sigma_beta, tau) for Sigma_alpha in covariance_matrices]
        S = torch.zeros((d, d), device = device).type(torch.complex64)
        for k in range(n):
            S += geometric_mean(inv(Sigma_beta), covariance_matrices_new[k])
        S = 1 / n * S
        S = (1-eta)* torch.eye(d, device = device) + eta * S
        Sigma_beta = torch.mm(torch.mm(S,Sigma_beta),S)
        loss_dis = UOT_distance_toset(covariance_matrices, Sigma_beta, tau)
        loss_record.append(loss_dis)
    #print(loss_record)    
    return Sigma_beta, loss_record


if __name__ == "__main__":
    seed_number = 4444
    torch.manual_seed(seed_number)
    # Example Usage
    num_samples = 1000
    num_dimensions = 50
    num_gaussians = 50
    sigma = 0.35

    gaussians, covariance_matrices, means = generate_centered_gaussians(num_samples, num_dimensions, num_gaussians)

    # Convert the data to PyTorch tensors
    covariance_matrices_torch = [torch.tensor(covariance_matrix, device=device) for covariance_matrix in covariance_matrices]
    beta_cov_torch = torch.eye(num_dimensions, device=device)

    # Initial guess for the barycenter
    Sigma_beta_0_torch = torch.eye(num_dimensions, device=device)

    # Compute the Bures UOT barycenter using PyTorch
    # result_matric_1, loss_1 = Bures_manifold_UOT_bary_gradientmethod_exactgrad_SGD(covariance_matrices_torch, Sigma_beta_0_torch, epsilon = 1e-6, T = 200, eta = 1, early_stop = True)
    # result_matric_2, loss_2 = Bures_manifold_UOT_bary_gradientmethod_autograd_SGD(covariance_matrices_torch, Sigma_beta_0_torch, epsilon = 1e-6, T = 200, eta = 1, early_stop = True)
    # result_matric_3, loss_3 = Bures_manifold_UOT_bary_gradientmethod_autograd_momentum_SGD(covariance_matrices_torch, Sigma_beta_0_torch, epsilon = 1e-6, T = 200, eta = 1, early_stop = True)
    # result_matric_4, loss_4 = Bures_manifold_UOT_bary_hybridmethod_SGD(covariance_matrices_torch, Sigma_beta_0_torch, epsilon = 1e-6, T = 200, eta = 1, early_stop = True)

    result_matric_1, loss_1 = Bures_manifold_UOT_bary_gradientmethod_exactgrad(covariance_matrices_torch, Sigma_beta_0_torch, epsilon=1e-6, T=250, eta=0.25, early_stop=True)
    result_matric_2, loss_2 = Bures_manifold_UOT_bary_gradientmethod_autograd(covariance_matrices_torch, Sigma_beta_0_torch, epsilon=1e-6, T=250, eta=0.25, early_stop=True)
    result_matric_3, loss_3 = Bures_manifold_UOT_bary_gradientmethod_autograd_momentum(covariance_matrices_torch, Sigma_beta_0_torch, T = 250, epsilon = 1e-6, eta = 0.25, early_stop = True)
    result_matric_4, loss_4 = Bures_manifold_UOT_bary_hybridmethod(covariance_matrices_torch, Sigma_beta_0_torch, T = 250, epsilon = 1e-6, eta = 1, early_stop = True)    

    loss_1 = [tensor.detach().cpu() for tensor in loss_1]
    loss_2 = [tensor.detach().cpu() for tensor in loss_2]
    loss_3 = [tensor.detach().cpu() for tensor in loss_3]
    loss_4 = [tensor.detach().cpu() for tensor in loss_4]

    # Create a list of iteration numbers for x-axis
    iterations_1 = list(range(1, len(loss_1) + 1))
    iterations_2 = list(range(1, len(loss_2) + 1))
    iterations_3 = list(range(1, len(loss_3) + 1))
    iterations_4 = list(range(1, len(loss_4) + 1))


    # Determine the maximum iteration value to set the x-axis limits
    max_iterations = max(len(loss_1), len(loss_2), len(loss_3), len(loss_4))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_1, loss_1, marker='s', label='Exact Geodesic GD')
    plt.plot(iterations_2, loss_2, marker='s', label='Auto GD')
    plt.plot(iterations_3, loss_3, marker='s', label='Auto GD + Momentum')
    plt.plot(iterations_4, loss_4, marker='s', label='Hybrid GD')

    # Labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    #plt.legend()

    # Set x-axis to contain only integer ticks
    xticks = list(range(0, (max_iterations // 10 + 1) * 10, 20))
    plt.xticks(xticks)

    # Display the plot
    plt.grid(True)
    plt.savefig(f"Image/rate_tau_0.01_dimension_{num_dimensions}.pdf")
    plt.show()
