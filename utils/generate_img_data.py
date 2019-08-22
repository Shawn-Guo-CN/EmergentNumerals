from PIL import Image, ImageDraw
import os
from utils.conf import args

SIZE_SPACE = 20 # size of the space for a single object

def generate_image(digits, base_img, obj_imgs, out_file, max_num_objs=5):
    assert len(digits) <= len(obj_imgs)

    cur_base = base_img.copy()
    for i in range(len(digits)):
        cur_num = int(digits[i])
        for j in range(cur_num):
            cur_base.paste(obj_imgs[i], (SIZE_SPACE*j + max_num_objs, SIZE_SPACE*i + max_num_objs))
    cur_base.save(out_file, 'PNG')

def generate_all_combinations(base_img, obj_imgs, prefix='', type_idx=0, out_dir='data/img_set_25', max_num_objs=5):
    if type_idx == args.num_words - 1:
        for i in range(0, args.max_len_word+1):
            target_str = str(i)
            print(prefix+target_str)
            generate_image(
                prefix+target_str, 
                base_img,
                obj_imgs, 
                out_file=os.path.join(out_dir, prefix+target_str+'.png'), 
                max_num_objs=max_num_objs
            )
    else:
        for i in range(0, args.max_len_word+1):
            target_str = str(i)
            generate_all_combinations(base_img, obj_imgs, prefix+target_str, type_idx+1, out_dir=out_dir)

def generate_image_set(
    base_img_path='data/base_imgs/background.png',
    obj_img_dir='data/base_imgs/',
    num_obj_types=2, # numbers of different types of objects
    max_num_objs=5, # maximum numbers of a specific kind of object
    out_dir='data/img_set_25'
):
    base_img = Image.open(base_img_path)

    obj_file_names = []
    for name in os.listdir(obj_img_dir):
        if name.startswith('obj_') and name.endswith('.png'):
            obj_file_names.append(name)

    obj_imgs = [Image.open(os.path.join(obj_img_dir+name)) for name in obj_file_names]

    generate_all_combinations(base_img, obj_imgs, out_dir='data/img_set_25', max_num_objs=max_num_objs)

    

if __name__ == '__main__':
    generate_image_set()