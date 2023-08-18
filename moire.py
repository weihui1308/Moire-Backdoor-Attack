import numpy as np
from PIL import Image
import cv2
import random


def stretch(tx, ty, amount, direction, img):
    if direction == 'horizontal':
        return tx + amount * tx, ty
    elif direction == 'vertical':
        return tx, ty + amount * (ty - img.shape[0] / 2)
    else:
        raise ValueError('Invalid direction')


def twist(tx, ty, amount, direction, img):
    if direction == 'clockwise':
        theta = amount * np.pi / 180
    elif direction == 'counterclockwise':
        theta = -amount * np.pi / 180
    else:
        raise ValueError('Invalid direction')
    cx, cy = img.shape[1] / 2, img.shape[0] / 2
    dx = tx - cx
    dy = ty - cy
    tx = cx + np.cos(theta) * dx - np.sin(theta) * dy
    ty = cy + np.sin(theta) * dx + np.cos(theta) * dy
    return tx, ty


def grid_warp(tx, ty, amount, frequency, img):
    cx, cy = img.shape[1] / 2, img.shape[0] / 2
    dx = tx - cx
    dy = ty - cy
    x_offset = amount * np.sin(dx * frequency)
    y_offset = amount * np.sin(dy * frequency)
    tx = tx + x_offset
    ty = ty + y_offset
    return tx, ty
    

class Moire:
    def __init__(self):
        self.moire = ['parallel_line_moire', 'rotated_line_moire', 'curved_line_moire']
        
    def __call__(self, masks, img, moire_type):
        w, h = img.shape[1], img.shape[0]
        long_side = w if w > h else h
        if moire_type == self.moire[0]:
            distence1 = random.randint(4, 7)
            distence2 = distence1 + random.randint(1, 4)
            result = parallel_line_moire(masks, img, distence1, distence2)
        elif moire_type == self.moire[1]:
            distence = random.randint(4, 7)
            angle = random.randint(7, 13)
            result = rotated_line_moire(masks, img, distence, angle)
        else:
            num = random.randint(int((long_side//2) / 7), int((long_side//2) / 3))
            distance = random.randint(4, 7)
            result = curved_line_moire(masks, img, num, distance)
        return result



def parallel_line_moire(masks, img, distence1, distence2):
    img = img.astype(np.float32) / 255.0
    w, h = img.shape[1], img.shape[0]
    moire = np.zeros((h, w))

    num1 = int(h / distence1)
    for i in range(num1):
        moire[i*distence1, :] = -1   

    tx, ty = np.meshgrid(np.arange(w), np.arange(h))
    tx, ty = stretch(tx, ty, amount=random.uniform(0, 0.5), direction=random.choice(['vertical', 'horizontal']), img=img)
    tx, ty = twist(tx, ty, amount=random.uniform(0, 180), direction=random.choice(['clockwise', 'counterclockwise']), img=img)
    tx, ty = grid_warp(tx, ty, amount=random.uniform(0, 1.5), frequency=random.uniform(0, 0.1), img=img)
    
    deformed_moire = cv2.remap(moire, tx.astype(np.float32), ty.astype(np.float32), cv2.INTER_LINEAR)
    
    num2 = int(h / distence2)
    for i in range(num2):
        deformed_moire[i*distence2, :] = -1
    
    deformed_moire = rotate_array(deformed_moire, angle=random.uniform(0, 180))
    
    moire_pattern_tmp = np.zeros_like(deformed_moire)
    for idx in range(masks.shape[0]):
        moire_pattern_tmp += (deformed_moire * masks[idx])
    
    result = np.zeros_like(img)
    for i in range(3):
        result[:, :, i] = img[:, :, i] + moire_pattern_tmp * 0.2

    return result


def rotated_line_moire(masks, img, distence, angle):
    img = img.astype(np.float32) / 255.0
    w, h = img.shape[1], img.shape[0]
    moire = np.zeros((h, w))
    
    num1 = int(h / distence)
    for i in range(num1):
        moire[i*distence, :] = -1
    
    moire = rotate_array(moire, angle)
    
    tx, ty = np.meshgrid(np.arange(w), np.arange(h))
    tx, ty = stretch(tx, ty, amount=random.uniform(0, 0.5), direction=random.choice(['vertical', 'horizontal']), img=img)
    tx, ty = twist(tx, ty, amount=random.uniform(0, 180), direction=random.choice(['clockwise', 'counterclockwise']), img=img)
    tx, ty = grid_warp(tx, ty, amount=random.uniform(0, 1.5), frequency=random.uniform(0, 0.1), img=img)
    
    deformed_moire = cv2.remap(moire, tx.astype(np.float32), ty.astype(np.float32), cv2.INTER_LINEAR)
    
    num2 = int(h / distence)
    for i in range(num2):
        deformed_moire[i*distence, :] = -1
    
    deformed_moire = rotate_array(deformed_moire, angle=random.uniform(0, 180))
    
    moire_pattern_tmp = np.zeros_like(deformed_moire)
    for idx in range(masks.shape[0]):
        moire_pattern_tmp += (deformed_moire * masks[idx])
    
    result = np.zeros_like(img)
    for i in range(3):
        result[:, :, i] = img[:, :, i] + moire_pattern_tmp * 0.2
        
    return result
    
    
def curved_line_moire(masks, img, num, distance):
    img = img.astype(np.float32) / 255.0
    w, h = img.shape[1], img.shape[0]
    long_side = w if w > h else h
    moire = np.zeros((h, w))
    
    center = (w // 2, h // 2)
    color = -1
    thickness = 1
    space = int((long_side//2) / num)
    for i in range(num+1):
        radius = space * i
        cv2.circle(moire, center, radius, color, thickness)
    
    tx, ty = np.meshgrid(np.arange(w), np.arange(h))
    tx, ty = stretch(tx, ty, amount=random.uniform(0, 0.5), direction=random.choice(['vertical', 'horizontal']), img=img)
    tx, ty = twist(tx, ty, amount=random.uniform(0, 180), direction=random.choice(['clockwise', 'counterclockwise']), img=img)
    tx, ty = grid_warp(tx, ty, amount=random.uniform(0, 1.5), frequency=random.uniform(0, 0.1), img=img)
    
    deformed_moire = cv2.remap(moire, tx.astype(np.float32), ty.astype(np.float32), cv2.INTER_LINEAR)
    
    for i in range(int(h / distance)):
        deformed_moire[i*distance, :] = -1
    
    deformed_moire = rotate_array(deformed_moire, angle=random.uniform(0, 180))
    
    moire_pattern_tmp = np.zeros_like(deformed_moire)
    for idx in range(masks.shape[0]):
        moire_pattern_tmp += (deformed_moire * masks[idx])
    
    result = np.zeros_like(img)
    for i in range(3):
        result[:, :, i] = img[:, :, i] + moire_pattern_tmp * 0.2
    
    return result
    

def rotate_array(array, angle):
    theta = np.radians(angle)
    x, y = np.array(array.shape) / 2
    x_indices, y_indices = np.indices(array.shape)
    x_indices = x_indices - x
    y_indices = y_indices - y
    x_rotated = x_indices * np.cos(theta) + y_indices * np.sin(theta)
    y_rotated = -x_indices * np.sin(theta) + y_indices * np.cos(theta)
    x_rotated = x_rotated + x
    y_rotated = y_rotated + y
    rotated_array = np.zeros(array.shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if 0 <= x_rotated[i][j] < array.shape[0] and 0 <= y_rotated[i][j] < array.shape[1]:
                rotated_array[i][j] = array[int(x_rotated[i][j])][int(y_rotated[i][j])]
    return rotated_array
    
    
    
    
    
    
    
    
    
    
    
    