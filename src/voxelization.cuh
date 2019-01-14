#ifndef VOXELIZATION_CUH
#define VOXELIZATION_CUH
int sparse_voxelization(uint64_t *keys, int *labels, int **return_key_indices,
                        int **return_labels, int n, int ignore_label,
                        int has_label);

void cuda_thread_exit(void);
#endif
