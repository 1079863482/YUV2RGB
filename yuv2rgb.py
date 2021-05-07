import os
import cv2
import numpy as np

IMG_WIDTH = 1152
IMG_HEIGHT = 648
IMG_SIZE = int(IMG_WIDTH * IMG_HEIGHT * 3 / 2)

Y_WIDTH = IMG_WIDTH
Y_HEIGHT = IMG_HEIGHT
Y_SIZE = int(Y_WIDTH * Y_HEIGHT)

U_V_WIDTH = int(IMG_WIDTH / 2)
U_V_HEIGHT = int(IMG_HEIGHT / 2)
U_V_SIZE = int(U_V_WIDTH * U_V_HEIGHT)


def from_I420(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_start = y_start + Y_SIZE
        v_start = u_start + U_V_SIZE
        v_end = v_start + U_V_SIZE

        Y[frame_idx, :, :] = yuv_data[y_start : u_start].reshape((Y_HEIGHT, Y_WIDTH))
        U[frame_idx, :, :] = yuv_data[u_start : v_start].reshape((U_V_HEIGHT, U_V_WIDTH))
        V[frame_idx, :, :] = yuv_data[v_start : v_end].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V

def from_YV12(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        v_start = y_start + Y_SIZE
        u_start = v_start + U_V_SIZE
        u_end = u_start + U_V_SIZE

        Y[frame_idx, :, :] = yuv_data[y_start : v_start].reshape((Y_HEIGHT, Y_WIDTH))
        V[frame_idx, :, :] = yuv_data[v_start : u_start].reshape((U_V_HEIGHT, U_V_WIDTH))
        U[frame_idx, :, :] = yuv_data[u_start : u_end].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def from_NV12(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_v_start = y_start + Y_SIZE
        u_v_end = u_v_start + (U_V_SIZE * 2)

        Y[frame_idx, :, :] = yuv_data[y_start : u_v_start].reshape((Y_HEIGHT, Y_WIDTH))
        U_V = yuv_data[u_v_start : u_v_end].reshape((U_V_SIZE, 2))
        U[frame_idx, :, :] = U_V[:, 0].reshape((U_V_HEIGHT, U_V_WIDTH))
        V[frame_idx, :, :] = U_V[:, 1].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def from_NV21(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_v_start = y_start + Y_SIZE
        u_v_end = u_v_start + (U_V_SIZE * 2)

        Y[frame_idx, :, :] = yuv_data[y_start : u_v_start].reshape((Y_HEIGHT, Y_WIDTH))
        U_V = yuv_data[u_v_start : u_v_end].reshape((U_V_SIZE, 2))
        V[frame_idx, :, :] = U_V[:, 0].reshape((U_V_HEIGHT, U_V_WIDTH))
        U[frame_idx, :, :] = U_V[:, 1].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V

def np_yuv2rgb(Y,U,V):
    bgr_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    V = np.repeat(V, 2, 0)
    V = np.repeat(V, 2, 1)
    U = np.repeat(U, 2, 0)
    U = np.repeat(U, 2, 1)

    c = (Y-np.array([16])) * 298
    d = U - np.array([128])
    e = V - np.array([128])

    r = (c + 409 * e + 128) // 256
    g = (c - 100 * d - 208 * e + 128) // 256
    b = (c + 516 * d + 128) // 256

    r = np.where(r < 0, 0, r)
    r = np.where(r > 255,255,r)

    g = np.where(g < 0, 0, g)
    g = np.where(g > 255,255,g)

    b = np.where(b < 0, 0, b)
    b = np.where(b > 255,255,b)

    bgr_data[:, :, 2] = r
    bgr_data[:, :, 1] = g
    bgr_data[:, :, 0] = b

    return bgr_data

def yuv2rgb(Y, U, V):
    bgr_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    for h_idx in range(Y_HEIGHT):
        for w_idx in range(Y_WIDTH):
            y = Y[h_idx, w_idx]
            u = U[int(h_idx // 2), int(w_idx // 2)]
            v = V[int(h_idx // 2), int(w_idx // 2)]

            c = (y - 16) * 298
            d = u - 128
            e = v - 128

            r = (c + 409 * e + 128) // 256
            g = (c - 100 * d - 208 * e + 128) // 256
            b = (c + 516 * d + 128) // 256

            bgr_data[h_idx, w_idx, 2] = 0 if r < 0 else (255 if r > 255 else r)
            bgr_data[h_idx, w_idx, 1] = 0 if g < 0 else (255 if g > 255 else g)
            bgr_data[h_idx, w_idx, 0] = 0 if b < 0 else (255 if b > 255 else b)

    return bgr_data

if __name__ == '__main__':
    import time

    yuv = "./test.yuv"
    frames = int(os.path.getsize(yuv) / IMG_SIZE)

    with open(yuv, "rb") as yuv_f:
        time1 = time.time()
        yuv_bytes = yuv_f.read()
        yuv_data = np.frombuffer(yuv_bytes, np.uint8)

        # Y, U, V = from_I420(yuv_data, frames)
        # Y, U, V = from_YV12(yuv_data, frames)
        # Y, U, V = from_NV12(yuv_data, frames)
        Y, U, V = from_NV21(yuv_data, frames)

        rgb_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        for frame_idx in range(frames):
            # bgr_data = yuv2rgb(Y[frame_idx, :, :], U[frame_idx, :, :], V[frame_idx, :, :])            # for
            bgr_data = np_yuv2rgb(Y[frame_idx, :, :], U[frame_idx, :, :], V[frame_idx, :, :])           # numpy
            time2 = time.time()
            print(time2 - time1)
            if bgr_data is not None:
                cv2.imwrite("frame_{}.jpg".format(frame_idx), bgr_data)
                frame_idx +=1
