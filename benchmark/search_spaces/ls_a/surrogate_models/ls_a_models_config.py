ls_a_models_config = {
    "gcn": {
        'emb_dim': 1050,
        'layer_num': 8,
        'qubit_num': 4,
        'num_node_features': 8,
        'drop_ratio': 0.0644893118913786,
        'batch_size': 32,
        'epochs': 100,
        'lr': 4.540520885756229e-05,
        'decay': 1.917208797826118e-06,
        'JK': 'mean',
        'graph_pooling': 'attention',
        'metric': 'spearman',
    },
    "random_forest": {
        'n_estimators': 325,
        'max_depth': 30,
        'min_samples_split': 3,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'batch_size': 64,
        'metric': 'spearman',
    }
}
