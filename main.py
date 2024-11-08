import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import sys
from pyvirtualdisplay import Display

#　ディスプレイを追加（今回のみ）
display = Display(visible=0, size=(1024, 768))
display.start()

# 画像の上端と下端の座標を検出する関数
def topbottom(img):
    img = (img > 128) * 255  # 画像を二値化
    
    rows, cols = img.shape  # 画像の行数と列数を取得

    # 上端の座標を取得
    for i in range(rows):
        for j in range(cols - 1, -1, -1):
            if img[i, j] == 255:
                img_top = (i, j)
                break
        if 'img_top' in locals():
            break

    # 上端検出の補足処理
    img_top_list = []
    img_y = img_top[0] + 10
    for j in range(cols - 1, -1, -1):
        if img[img_y, j] == 255:
            img_top_list.append(j)
            break
    for j in range(cols):
        if img[img_y, j] == 255:
            img_top_list.append(j)
            break
    img_top = [img_y, np.average(img_top_list)]

    # 下端の座標を取得
    for i in range(rows - 1, -1, -1):
        for j in range(cols):
            if img[i, j] == 255:
                img_bottom = (i, j)
                break
        if 'img_bottom' in locals():
            break

    return img_top, img_bottom

# 画像を正方形に拡大・縮小する関数
def image_press(or_im):
    im = or_im

    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    im_new = expand2square(im, (255, 255, 255))
    return im_new.resize((416, 416))

# StreamlitでのUI構築
st.title("画像アップロードまたはカメラで撮影")

# 画像のアップロードとカメラ入力
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("またはカメラで画像を撮影してください")

# アップロードされた画像があれば処理を開始
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif camera_image is not None:
    image = Image.open(camera_image)

# 画像の前処理と物体検出の実行
if 'image' in locals():
    org_img = image_press(image)  # 画像の前処理
    img_size = 300

    # 物体検出ボタンが押された場合
    if st.button('物体検出', key='my_button'):
        # モデルをロードし、予測を実行
        model = YOLO("e_meter_segadd2.pt")
        results = model.predict(org_img, imgsz=416, conf=0.5, classes=0)

        # 検出結果の確認
        if results[0].masks is None or len(results[0].masks) < 3:
            st.error("配線の検出に失敗しました。目視で確認してください", icon="🚨")
            st.image(org_img, width=img_size)
            sys.exit("配線の検出に失敗しました。目視で確認してください")

        # 結果の格納用辞書
        processed_data = {
            'images': [],
            'coordinates': [],
            'classification': ''
        }

        # 各マスク画像の処理
        for r in results[0].masks:
            mask_img = (r.data[0].cpu().numpy() * 255).astype(int)
            top, bottom = topbottom(mask_img)
            processed_data['images'].append(mask_img)
            processed_data['coordinates'].append((top, bottom))

        # 座標を取得し、上端と下端のリストを作成
        coordinates_list = processed_data['coordinates']
        connect_list = np.array(coordinates_list)
        
        top_list = [coord[0] for coord in connect_list]
        bottom_list = [coord[1] for coord in connect_list]

        # 3本の配線検出時の処理
        if len(top_list) == 3:
            y0, x0 = zip(*sorted(top_list, key=lambda x: x[1]))
            xl1, xl2 = x0[1] - x0[0], x0[2] - x0[1]

            # 欠損推定と仮定位置の計算
            width = min(xl1, xl2)
            topx_dummy = x0[2] - width if xl1 < xl2 else x0[0] + width
            y_avr = int(sum(y0) / len(y0))
            top_list.append((y_avr, topx_dummy))

            # 中心座標の算出
            connect_center = sorted(top_list, key=lambda x: x[1])[1][1]

            # 下端データの補完と振り分け
            y0, x0 = zip(*sorted(bottom_list, key=lambda x: x[1]))
            y_avr = int(sum(y0) / len(y0))
            btmx_dummy = x0[0] - 5 if len(x0) == 1 else x0[-1] + 5
            bottom_list.append((y_avr, btmx_dummy))

        # 上端と下端のソートと分類判定
        sorted_top = np.array(sorted(top_list, key=lambda x: x[1]))
        sorted_bottom = np.array([tup for _, tup in sorted(zip(top_list, bottom_list), key=lambda x: x[0][1])])
        center = sum([coord[1] for coord in sorted_bottom]) / 4

        # 正しい配線の判定
        if np.all(sorted_bottom[::2, 1] < center) and np.all(sorted_bottom[1::2, 1] > center):
            st.success("左電源の正結線の可能性が高いです", icon="✅")
        elif np.all(sorted_bottom[::2, 1] > center) and np.all(sorted_bottom[1::2, 1] < center):
            st.success("右電源の正結線の可能性が高いです", icon="✅")
        else:
            st.warning("誤配線の可能性があります。目視で確認してください", icon="⚠")

        # 検出結果の可視化
        im = Image.fromarray(results[0].plot(boxes=False)[..., ::-1])
        draw = ImageDraw.Draw(im)
        for y, x in sorted_top:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 255))
        for y, x in sorted_bottom:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 0, 255))

        st.image(im, width=img_size)

    st.write("検出前")
    st.image(org_img, width=img_size)
