from PIL import Image
import numpy as np

front_path = "res/3_res.jpg"
back_path = "portrait-back/maple.jpg"
save_path = "res/maple1.jpg"

# w_dim = 256
# h_dim = 256
back_w = 0.5
back_h = 0.5
front_w = 1
front_h = 1

back = Image.open(back_path)
tmp = np.array(back)
back = back.resize((int(tmp.shape[1]*back_w), int(tmp.shape[0]*back_h)))
back = np.array(back)

front = Image.open(front_path)
tmp = np.array(front)
w_dim, h_dim, _ = tmp.shape

front = front.resize((int(h_dim*front_h), int(w_dim*front_w)))
front = np.array(front)

#back[-w_dim:, -h_dim:, :][front!=0] = 0
back[int(-w_dim*front_w):, int(-h_dim*front_h):, :][front>15] = front[front>15]

# back[-400:, -400:, :] *= front

res_img = Image.fromarray(back)

res_img.save(save_path)
