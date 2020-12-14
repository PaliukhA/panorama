from panorama import ransac_transform
from numpy import load, mean
from numpy.linalg import norm


def test_ransac():
    # test for points without outliers
    data = load('./tests/data/ransac_test1.npz')
    src_keypoints = data['src_k']
    dest_keypoints = data['dest_k']
    src_descriptors = data['src_d']
    dest_descriptors = data['dest_d']
    transform, matches = ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors,
                                          return_matches=True)
    print(matches)
    assert abs(len(matches) - len(src_keypoints)) < 3
    # assert(False)


def test_ransac_2():
    # test for correct projections of matches
    threshold = 1.5
    data = load('./tests/data/ransac_test2.npz')
    src_keypoints = data['src_k']
    dest_keypoints = data['dest_k']
    src_descriptors = data['src_d']
    dest_descriptors = data['dest_d']
    print(type(dest_descriptors[0]))
    transform, matches = ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors,
                                          return_matches=True)
    # print(src_keypoints)
    # print(dest_keypoints)
    print(matches)
    # print("result=", dest_keypoints[matches[:, 1]])
    # print(transform(src_keypoints[matches[:, 0]]))
    assert mean(norm(dest_keypoints[matches[:, 1]] - transform(src_keypoints[matches[:, 0]]), axis=1)) < threshold
    # assert(False)

def test_ransac_3():
    # test for correct projections of matches
    threshold = 2.5
    data = load('./tests/data/ransac_test3.npz')
    src_keypoints = data['src_k']
    dest_keypoints = data['dest_k']
    src_descriptors = data['src_d']
    dest_descriptors = data['dest_d']
    print(type(dest_descriptors[0]))
    transform, matches = ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors,
                                          return_matches=True)
    print(matches)
    assert mean(norm(dest_keypoints[matches[:, 1]] - transform(src_keypoints[matches[:, 0]]), axis=1)) < threshold
    # assert(False)