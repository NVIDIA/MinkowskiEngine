# Minkowski Engine Examples


## ModelNet40 Classification

```
python -m examples.classification_modelnet40 --network pointnet  # torch PointNet
python -m examples.classification_modelnet40 --network minkpointnet  # MinkowskiEngine PointNet
python -m examples.classification_modelnet40 --network minkfcnn  # MinkowskiEngine FCNN
```

### Training Logs

- training log for MinkowskiFCNN: [https://pastebin.pl/view/30f0a0c8](https://pastebin.pl/view/30f0a0c8)

## ScanNet Semantic Segmentation

```
python -m examples.indoor
```
