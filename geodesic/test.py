import argparse
import h5py
import numpy as np
from geodesic import GeodesicDistanceComputation


def main(input_animation_file, output_sploc_file):
    with h5py.File(input_animation_file, 'r') as f:
        verts = f['verts'].value.astype(np.float)
        tris = f['tris'].value
    N, _ = verts.shape
    compute_distance = GeodesicDistanceComputation(verts, tris)

    with h5py.File(output_sploc_file, 'w') as f:
        f['Gnum'] = N
        for i in range(0, N):
            f['geodis%05d' % i] = compute_distance(i)
#main('F:\\yangjiee\\yangjie\\tracking\\paper\\_tem_\\inputfine.h5','F:\\yangjiee\\yangjie\\tracking\\paper\\_tem_\\inputfine.h5')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find Sparse Localized Deformation Components')
    parser.add_argument('input_animation_file')
    parser.add_argument('output_sploc_file')
    args = parser.parse_args()
    main(args.input_animation_file,
         args.output_sploc_file)
