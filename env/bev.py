import cv2
import numpy as np
import math

def get_intrinsic_matrix(offset_x=0, offset_y=0, offset_z=25):
    # camera matrix
    K = np.array([
        [1173.122620, 0.000000, 969.335924 + offset_x, 0],
        [0.000000, 1179.612539, 549.524382 + offset_y, 0],
        [0.000000, 0.000000, 1.000000, 0]
    ])
    S = np.array([
        [96/1920.0, 0, 0],
        [0, (96-offset_z)/1088.0, 0],
        [0, 0, 1]
    ])
    K = np.matmul(S, K)
    return K


def get_transformation_matrix(K):
    DISTANCE_FACTOR = 0.5
    M = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, K[0, 0] * DISTANCE_FACTOR],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # translation matrix
    OFFSET = 10
    T = np.array([
        [1, 0, -K[0, 2]],
        [0, 1, -K[1, 2] - OFFSET],
        [0, 0, 0],
        [0, 0, 1]
    ])

    # rotation matrix
    SLOPE_FACTOR = 0.2
    alpha = -math.pi/(2 + SLOPE_FACTOR)
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha), 0],
        [0, np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 0, 1]
    ])

    # extrinsic matrix
    M = np.matmul(M, np.matmul(Rx, T))
    return M

# homography matrix
def get_homography_matrix(offset_x=0., offset_y=0., offset_z=25):
    K = get_intrinsic_matrix(offset_x=offset_x, offset_y=offset_y, offset_z=offset_z)
    M = get_transformation_matrix(K=K)
    H = np.matmul(K, M)
    return H

# change screen form bird eye view to camera on the car view
def from_bird_view(screen, offset_x=50, offset_y=0., offset_z=25):
    orig_w, orig_h, _ = screen.shape

    # get homography matrix
    H = get_homography_matrix(offset_x, offset_y, offset_z)

    # RGB to BGR and crop
    screen = screen[:, :, ::-1]
    screen = screen[:-offset_z, :, ::-1]

    w, h, _ = screen.shape
    road2 = cv2.warpPerspective(screen, H, (h, w), flags=cv2.INTER_CUBIC)
    road2 = cv2.resize(road2, (orig_w, orig_h))

    # segmentation
    R, G, B = screen[:, :, 0], screen[:, :, 1], screen[:, :, 2]
    road = ((0.35 < R/255.) * (R/255.< 0.45) * (0.35 < G/255.) * (G/255. < 0.45) * (0.35 < B/255.0) *(B/255.0 < 0.45)).astype(np.uint8)
    road = 255 * road

    # apply close
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((5,5)))

    # change perspective
    w, h, _ = screen.shape
    road = cv2.warpPerspective(road, H, (h, w), flags=cv2.INTER_CUBIC)
    road = cv2.resize(road, (orig_w, orig_h))
    return road


if __name__ == "__main__":
    import gym
    env = gym.make("CarRacing-v0")
    print(env.action_space)


    observation = env.reset()
    for _ in range(10000):
        action = np.array([0, 0.1, 0])  # env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        road = from_bird_view(observation)
        cv2.imshow("Screen", road)
        cv2.waitKey(0)

        if done:
            env.reset()

    env.close()
