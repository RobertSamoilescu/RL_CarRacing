import cv2
import numpy as np
import math

# camera matrix
K = np.array([
    [1173.122620, 0.000000, 969.335924, 0],
    [0.000000, 1179.612539, 549.524382, 0],
    [0.000000, 0.000000, 1.000000, 0]
])
S = np.array([
    [96/1920.0, 0, 0],
    [0, (96-25)/1088.0, 0],
    [0, 0, 1]
])
K = np.matmul(S, K)

DISTANCE_FACTOR = 0.5
M = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, K[0, 0] * DISTANCE_FACTOR],
    [0.0, 0.0, 0.0, 1.0]
])

A = np.array([
    [1, 0, -K[0, 2]],
    [0, 1, -K[1, 2] - 10],
    [0, 0, 0],
    [0, 0, 1]
])

# rotation matrix
eps = 0.2
alpha = -math.pi/(2 + eps)
Rx = np.array([
    [1, 0, 0, 0],
    [0, np.cos(alpha), -np.sin(alpha), 0],
    [0, np.sin(alpha), np.cos(alpha), 0],
    [0, 0, 0, 1]
])

# homography matrix
H = np.matmul(np.matmul(K, M), np.matmul(Rx, A))


# remove small connected compontents
def remove_small_cc(mask):
    # connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 1500

    # your answer image
    for j in range(0, nb_components):
        if sizes[j] <= min_size:
            mask[output == j + 1] = 0
    return mask


# change screen form bird eye view to camera on the car view
def from_bird_view(screen):
    orig_w, orig_h, _ = screen.shape

    # RGB to BGR and crop
    screen = screen[:, :, ::-1]
    screen = screen[:-25, :, ::-1]

    w, h, _ = screen.shape
    road2 = cv2.warpPerspective(screen, H, (h, w), flags=cv2.INTER_CUBIC)
    road2 = cv2.resize(road2, (orig_w, orig_h))

    # segmentation
    R, G, B = screen[:, :, 0], screen[:, :, 1], screen[:, :, 2]
    # grass = (G > 150).astype(np.uint8)
    # grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((5, 5)))
    # road = 255 * (1 - grass)
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
