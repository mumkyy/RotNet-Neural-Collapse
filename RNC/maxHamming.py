import numpy as np
from itertools import permutations
from typing import List, Tuple


def hamming_distance(p1: np.ndarray, p2: np.ndarray) -> int:
    """Calculate Hamming distance between two permutations."""
    return np.sum(p1 != p2)


def hamming_distance_matrix(P: List[Tuple], P_prime: List[Tuple]) -> np.ndarray:
    """
    Calculate Hamming distance matrix between two sets of permutations.
    
    Args:
        P: List of permutations (current set)
        P_prime: List of permutations (remaining candidates)
    
    Returns:
        Distance matrix of shape (len(P), len(P_prime))
    """
    D = np.zeros((len(P), len(P_prime)), dtype=int)
    for i, p1 in enumerate(P):
        for j, p2 in enumerate(P_prime):
            D[i, j] = hamming_distance(np.array(p1), np.array(p2))
    return D


def generate_maximal_hamming_distance_set(N: int, K: int) -> List[Tuple]:
    """
    Generate a maximal Hamming distance permutation set from a set of size K.
    
    Algorithm: Greedily select N permutations such that each new permutation
    maximizes the minimum Hamming distance to all previously selected permutations.
    
    Args:
        N: Number of permutations to select
    
    Returns:
        P: List of N permutations (maximal permutation set)
    """
    # Step 1: Generate all permutations of [1, 2, ..., 9]
    P_bar = list(permutations(range(1, K+1)))  # All 9! permutations
    
    # Step 2: Initialize output set P as empty
    P = []
    
    # Step 3: Randomly sample j from [1, 9!]
    j = np.random.randint(0, len(P_bar))
    
    # Step 4: i = 1
    i = 1
    
    # Step 5-12: Repeat until we have N permutations
    while i <= N:
        # Step 6: Add P_bar[j] to P
        P.append(P_bar[j])
        
        # Step 7: Remove P_bar[j] from P_bar
        P_prime = P_bar[:j] + P_bar[j+1:]
        
        if i < N:  # Only continue if we need more permutations
            # Step 8: Calculate Hamming distance matrix D
            D = hamming_distance_matrix(P, P_prime)
            
            # Step 9: D_bar = 1^T D (column sums / min distance to any selected perm)
            # Actually, we want the minimum distance for each candidate
            D_bar = np.min(D, axis=0)
            
            # Step 10: j = arg max_k D_bar[k]
            j = np.argmax(D_bar)
        
        # Update P_bar for next iteration
        P_bar = P_prime
        
        # Step 11: i = i + 1
        i += 1
    
    return P


def visualize_set(P: List[Tuple], title: str = "Maximal Hamming Distance Set"):
    """Visualize the permutation set and its pairwise distances."""
    import matplotlib.pyplot as plt
    
    n = len(P)
    
    # Calculate pairwise distance matrix
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = hamming_distance(np.array(P[i]), np.array(P[j]))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Heatmap of distance matrix
    im = ax1.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xlabel('Permutation Index')
    ax1.set_ylabel('Permutation Index')
    ax1.set_title('Pairwise Hamming Distances')
    plt.colorbar(im, ax=ax1, label='Hamming Distance')
    
    # Plot 2: Distribution of pairwise distances
    upper_triangle = dist_matrix[np.triu_indices(n, k=1)]
    ax2.hist(upper_triangle, bins=range(0, 11), edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Hamming Distance')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Pairwise Distances')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, dist_matrix


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate a maximal set of 5 permutations
    N = 5
    P = generate_maximal_hamming_distance_set(N)
    
    print(f"Generated {len(P)} permutations:")
    for i, perm in enumerate(P, 1):
        print(f"{i}: {perm}")
    
    # Calculate statistics
    min_distances = []
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            dist = hamming_distance(np.array(P[i]), np.array(P[j]))
            min_distances.append(dist)
    
    print(f"\nMinimum pairwise distance: {min(min_distances)}")
    print(f"Maximum pairwise distance: {max(min_distances)}")
    print(f"Mean pairwise distance: {np.mean(min_distances):.2f}")
    
    # Visualize
    fig, dist_matrix = visualize_set(P, f"Maximal Hamming Distance Set (N={N})")
    plt.savefig('/mnt/user-data/outputs/maximal_set_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved!")
