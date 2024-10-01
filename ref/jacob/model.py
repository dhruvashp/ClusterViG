import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_kmeans import KMeans

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, d_model):
        super(PatchEmbedding, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embedding(x)  # [batch, d_model, H/P, W/P]
        patches = x.flatten(2)  # [batch, d_model, N_patches]
        patches = patches.transpose(1, 2)  # [batch, N_patches, d_model]
        return patches

class DynamicClustering(nn.Module):
    def __init__(self, n_clusters, d_model):
        super(DynamicClustering, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.kmeans = KMeans(n_clusters=n_clusters)

    def forward(self, patches):
        patches = self.layer_norm(patches)  # Normalize patches
        cluster_result = self.kmeans(patches)
        cluster_centers = cluster_result.centers
        soft_assignments = cluster_result.soft_assignment
        return cluster_centers, soft_assignments

class AttentionMechanism(nn.Module):
    def __init__(self, d_model, n_clusters):
        super(AttentionMechanism, self).__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(n_clusters, d_model))

    def forward(self, clusters):
        clusters += self.pos_encoding  # Add positional encoding
        Q = self.query_proj(clusters)
        K = self.key_proj(clusters)
        V = self.value_proj(clusters)
        attn_scores = torch.einsum('bqd,bkd->bqk', Q, K) / (self.d_model ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        cluster_output = torch.einsum('bqk,bkd->bqd', attn_weights, V)
        return cluster_output

class HypergraphLaplacianLayer(nn.Module):
    def __init__(self, d_model, num_hyperedges, num_nodes):
        super(HypergraphLaplacianLayer, self).__init__()
        self.num_hyperedges = num_hyperedges
        self.num_nodes = num_nodes
        self.edge_proj = nn.Linear(d_model, num_hyperedges)
        self.node_proj = nn.Linear(d_model, d_model)

    def compute_laplacian(self, hyperedge_weights):
        """
        Compute the Laplacian matrix for the hypergraph.
        - hyperedge_weights: Tensor of shape [batch, N_nodes, N_hyperedges]
        """
        # Degree matrix D (sum of edge weights for each node)
        D = torch.sum(hyperedge_weights, dim=1)  # [batch, N_hyperedges]

        # Incidence matrix H
        H = hyperedge_weights.transpose(1, 2)  # [batch, N_hyperedges, N_nodes]

        # Laplacian matrix L = I - D^(-1/2) H H^T D^(-1/2)
        D_inv_sqrt = torch.diag_embed(D.pow(-0.5))  # [batch, N_hyperedges, N_hyperedges]
        L = torch.eye(H.size(1)).to(H.device) - torch.matmul(torch.matmul(D_inv_sqrt, H), H.transpose(-1, -2))

        # Adjust size of L to match the number of nodes
        if L.size(1) != self.num_nodes:
            L = F.pad(L, (0, self.num_nodes - L.size(1), 0, self.num_nodes - L.size(2)))

        return L

    def forward(self, clusters):
        # Compute hyperedges (relations between clusters)
        hyperedge_logits = self.edge_proj(clusters)
        hyperedge_weights = F.softmax(hyperedge_logits, dim=1)  # [batch, N_nodes, N_hyperedges]
        
        # Compute Laplacian
        L = self.compute_laplacian(hyperedge_weights)

        # Apply Laplacian propagation
        clusters = torch.einsum('bij,bjd->bid', L, clusters)

        # Node update via non-linear propagation
        updated_clusters = self.node_proj(clusters)
        return updated_clusters

class NonLinearPropagation(nn.Module):
    def __init__(self, d_model):
        super(NonLinearPropagation, self).__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, clusters):
        return self.attn_mlp(clusters)

class AdaptiveNodeCollapse(nn.Module):
    def __init__(self, collapse_threshold=0.9):
        super(AdaptiveNodeCollapse, self).__init__()
        self.collapse_threshold = collapse_threshold

    def forward(self, clusters):
        norm_clusters = F.normalize(clusters, dim=-1)
        similarity_matrix = torch.einsum('bqd,bkd->bqk', norm_clusters, norm_clusters)
        collapse_mask = similarity_matrix > self.collapse_threshold

        for i in range(clusters.size(1)):
            for j in range(i + 1, clusters.size(1)):
                if collapse_mask[:, i, j].any():
                    clusters[:, i] = (clusters[:, i] + clusters[:, j]) / 2
                    clusters[:, j] = clusters[:, i]  # Merge nodes
        return clusters

class FeedbackModulationLayer(nn.Module):
    def __init__(self, d_model):
        super(FeedbackModulationLayer, self).__init__()
        self.fb_layer = nn.Linear(d_model, d_model)
        self.fb_gate = nn.Linear(d_model, 1)

    def forward(self, clusters):
        global_features = clusters.mean(dim=1)  # Global average pooling
        feedback = self.fb_layer(global_features)
        feedback_gate = torch.sigmoid(self.fb_gate(global_features))
        return clusters + feedback.unsqueeze(1) * feedback_gate.unsqueeze(1)

class Net(nn.Module):
    # def __init__(self, img_size=(224, 224), patch_size=16, k_clusters=16, d_model=512, num_classes=1000, num_hyperedges=8):
    def __init__(self, img_size=(32, 32), patch_size=2, k_clusters=16, d_model=128, num_classes=10, num_hyperedges=8):
    
        super(Net, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, d_model)
        self.dynamic_clustering = DynamicClustering(k_clusters, d_model)
        self.attention_mechanism = AttentionMechanism(d_model, k_clusters)
        
        # Pass k_clusters as num_nodes to HypergraphLaplacianLayer
        self.hypergraph_laplacian_layer = HypergraphLaplacianLayer(d_model, num_hyperedges, num_nodes=k_clusters)
        
        self.non_linear_propagation = NonLinearPropagation(d_model)
        self.adaptive_node_collapse = AdaptiveNodeCollapse()
        self.feedback_modulation = FeedbackModulationLayer(d_model)
        
        # Classifier head
        self.classifier = nn.Linear(d_model * k_clusters, num_classes)

    def forward(self, x):
        patches = self.patch_embedding(x)  # Step 1: Patch extraction, DP: B, N, D
        
        
        # Step 2: Dynamic clustering
        clusters, soft_assignments = self.dynamic_clustering(patches)       # DP: B, clusters, D
        
        # Step 3: Attention mechanism
        clusters = self.attention_mechanism(clusters)                       # DP: B, clusters, D'
        
        # Step 4: Hypergraph Laplacian propagation
        clusters = self.hypergraph_laplacian_layer(clusters)
        
        # Step 5: Non-linear propagation
        clusters = self.non_linear_propagation(clusters)
        
        # Step 6: Adaptive node collapse
        clusters = self.adaptive_node_collapse(clusters)
        
        # Step 7: Feedback modulation
        clusters = self.feedback_modulation(clusters)
        
        # Flatten cluster representations for classification
        clusters_flat = clusters.flatten(1)  # [batch, d_model * k_clusters]
        
        # Final classification layer
        logits = self.classifier(clusters_flat)
        
        return logits
