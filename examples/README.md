# Minkowski Engine Examples


## ModelNet40

This example provides a demo script for training 3D Sparse Tensor Network with the ResNet50 backbone.
For the first epoch, the script will cache all meshes into the memory for faster data loading and might be slow.

```bash
python -m examples.modelnet40 \
	--batch_size 256 \       # batch size
	--lr 1e-2 \              # learning rate
	--sample_density 2000 \  # mesh sampling density
	--voxel_size 0.02 \      # voxel size. Use higher sampling density with smaller voxel
	---max_iter 120000       # Number of iterations for training
```

This is a typical output that you can expect after the caching is complete on TitanXP.

```bash
cchoy-dt 08/14 17:40:16 Iter: 41700, Loss: 1.020e+00, Data Loading Time: 2.160e-02, Tot Time: 6.279e-01
cchoy-dt 08/14 17:40:48 Iter: 41750, Loss: 9.807e-01, Data Loading Time: 1.402e-02, Tot Time: 5.942e-01
cchoy-dt 08/14 17:41:20 Iter: 41800, Loss: 1.195e+00, Data Loading Time: 1.482e-02, Tot Time: 7.101e-01
cchoy-dt 08/14 17:41:51 Iter: 41850, Loss: 1.089e+00, Data Loading Time: 1.368e-02, Tot Time: 6.088e-01
cchoy-dt 08/14 17:42:24 Iter: 41900, Loss: 8.747e-01, Data Loading Time: 1.686e-02, Tot Time: 6.631e-01
cchoy-dt 08/14 17:42:56 Iter: 41950, Loss: 1.048e+00, Data Loading Time: 1.391e-02, Tot Time: 6.165e-01
cchoy-dt 08/14 17:43:27 Iter: 42000, Loss: 1.324e+00, Data Loading Time: 1.634e-02, Tot Time: 6.165e-01
cchoy-dt 08/14 17:43:30 Validation
cchoy-dt 08/14 17:43:30 val set iter: 0 / 4, Accuracy : 8.984e-01
cchoy-dt 08/14 17:43:31 val set accuracy : 8.818e-01
cchoy-dt 08/14 17:43:31 LR: [0.0012851215656510308]
```
