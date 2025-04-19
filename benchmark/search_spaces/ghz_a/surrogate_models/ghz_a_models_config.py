ghz_a_models_config = {
    "qcn": {
        'emb_dim': 1200,
        'layer_num': 8,
        'qubit_num': 3,
        'num_node_features': 7,
        'drop_ratio': 0.012714767230404513,
        'batch_size': 32,
        'epochs': 3,
        'lr': 0.00042048670814195114,
        'decay': 1.2239395743425164e-06,
        'JK': 'mean',
        'graph_pooling': 'max',
        'metric': 'spearman',
    },
    "random_forest": {
        'n_estimators': 350,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'batch_size': 64,
        'metric': 'mse',
    }
}
