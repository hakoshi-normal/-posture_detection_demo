import time
import os
import shutil
import csv
import base64
import eel
import cv2
import numpy as np
import copy
import pyrealsense2 as rs
import onnxruntime
# import tensorflow as tf
import mediapipe as mp

info = {"meter": 2.0,
        "tools": [True,False],
        "algs": [False, True, False, False, False],
        "kernel": np.ones((1,1),np.uint8),
        "depth_bool": True,
        "rec": False,
        "play": False,
        "project": "Project01",
        "past_name":"",
        "start_time":0,
        "f_counter":0}

# saveディレクトリ作成
try:
    os.mkdir(f'save')
except FileExistsError:
    pass

depth_scale=0.0010000000474974513
try:
    # パイプライン作成
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
except RuntimeError:
    info["depth_bool"] = False
    info["algs"][1] = False
    info["algs"][0] = True
    cap = cv2.VideoCapture(0)


def conv_depth():
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        pass

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return depth_image, color_image

def patch_mask(depth_image, color_image, clipping_distance):
    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    # Render images:
    #   depth align to color on left
    #   depth on right
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # images = np.hstack((bg_removed, depth_colormap))
    return bg_removed


# 背景差分
bgs_list = [cv2.createBackgroundSubtractorMOG2(),
            cv2.createBackgroundSubtractorKNN(),
            cv2.bgsegm.createBackgroundSubtractorCNT()]

def conv_backsub(img, bgs, kernel):
    mask = bgs.apply(img)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # img[mask == 0] = 0
    # bg = bgs.getBackgroundImage()
    img = cv2.bitwise_and(img, mask)
    return img




# mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

media_models=[]
for _ in range(len(info["algs"])):
    pose = mp_pose.Pose(
            min_tracking_confidence=0,
            min_detection_confidence=0)
    media_models.append(pose)

def media(image, i):
    #BG_COLOR = (192, 192, 192) # gray
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = media_models[i].process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    if not results.pose_landmarks:
        return None

    points = []
    ids = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]
    for id in ids:
        x = results.pose_landmarks.landmark[id].x * image_width
        y = results.pose_landmarks.landmark[id].y * image_height
        points.append([int(x),int(y)])
    return points



# movenet
keypoint_score_th = 0.1
model_path = "onnx/movenet_singlepose_lightning_4.onnx"
input_size = 192

move_models=[]
for _ in range(len(info["algs"])):
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            #'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ],
    )
    move_models.append(onnx_session)

def run_inference(onnx_session, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # 前処理
    input_image = cv2.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = input_image.astype('int32')  # int32へキャスト

    # 推論
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: input_image})

    keypoints_with_scores = outputs[0]
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # キーポイント、スコア取り出し
    keypoints = []
    scores = []

    for index in range(17):
        keypoint_x = image_width * keypoints_with_scores[index][1]
        keypoint_y = image_height * keypoints_with_scores[index][0]
        score = keypoints_with_scores[index][2]

        keypoints.append([int(keypoint_x), int(keypoint_y)])
        scores.append(score)

    return keypoints, scores


def draw_debug(debug_image, keypoints):
    image = copy.deepcopy(debug_image)

    # # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
    # # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
    # # Line：鼻 → 左目
    index_list = [[0,1],[0,2],[1,3],[2,4],[0,5],[0,6],[5,6],[5,7],[7,9],[6,8],[8,10],[11,12],[5,11],[11,13],[13,15],[6,12],[12,14],[14,16]]
    for index in index_list:
        index01, index02 = index[0],index[1]
        # if scores[index01] > keypoint_score_th and scores[
        #         index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(image, point01, point02, (255, 255, 255), 4)
        cv2.line(image, point01, point02, (0, 0, 0), 2)

    # Circle：各点
    # for keypoint, score in zip(keypoints, scores):
    for keypoint in keypoints:
    # if score > keypoint_score_th:
        cv2.circle(image, keypoint, 6, (255, 255, 255), -1)
        cv2.circle(image, keypoint, 3, (0, 0, 0), -1)
    return image


def renew_projects():
    info["past_name"]=""
    files = os.listdir('save')
    dirs = [f for f in files if os.path.isdir(os.path.join('save', f))]
    if len(dirs)==0:
        eel.set_dirs(["No Video"])
        info["project"] = "No Video"
        eel.play_disabled(1)
    else:
        eel.set_dirs(dirs)
        info["project"] = dirs[0]
        eel.play_disabled(0)

def timer_reset():
    info["f_counter"] = 0
    info["start"] = time.time()


def main(info, depth_scale):
    eel.init('web')

    @eel.expose
    def set_meters(x):
        info["meter"]=float(x)
        timer_reset()
    
    @eel.expose
    def set_tools(x):
        info["tools"][x] = not info["tools"][x]
        timer_reset()

    @eel.expose
    def set_algs(x):
        info["algs"][x] = not info["algs"][x]
        timer_reset()

    @eel.expose
    def set_kernelsize(x):
        x = int(x)
        info["kernel"] = np.ones((x,x),np.uint8)
        timer_reset()
    
    @eel.expose
    def set_rec():
        info["rec"] = not info["rec"]

    @eel.expose
    def set_play():
        info["play"] = not info["play"]
    
    @eel.expose
    def set_project(name):
        info["project"] = name

    @eel.expose
    def del_project(name):
        try:
            shutil.rmtree(f'save/{name}')
        except:
            pass
        renew_projects()

    eel.start(
        'index.html',
        mode='chrome',
        suppress_error=True,
        cmdline_args=['--start-fullscreen'],
        block=False)


    if not info["depth_bool"]:
        eel.depth_disabled()

    # プロジェクト一覧更新
    renew_projects()


    depth_images=[]
    color_images=[]
    rec_flg = False
    rec_counter = 0
    rec_timer = 0

    idx = 0
    clipping_distance = 0

    info["start"] = time.time()
    while True:
        eel.sleep(0.01)
        
        # 録画
        if info["rec"]:
            if not rec_flg: # 初回
                depth_images=[]
                color_images=[]
                rec_timer = time.time()
                rec_flg = True
            if info["depth_bool"]: # realsense
                depth_img, color_img = conv_depth()
                depth_data = copy.deepcopy(depth_img)
                color_data = copy.deepcopy(color_img)
            else:
                ret, color_data = cap.read()
                depth_data = np.zeros((480,640))
            depth_images.append(depth_data)
            color_images.append(color_data)

            rec_counter+=1
        elif not info["rec"] and rec_flg:
            fps = rec_counter/(time.time()-rec_timer)
            try:
                os.mkdir(f'save/{info["project"]}')
            except FileExistsError:
                pass
            with open(f'save/{info["project"]}/video_info.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow([depth_scale,fps])
            np.save(f'save/{info["project"]}/depth_save.npy', depth_images)
            np.save(f'save/{info["project"]}/color_save.npy', color_images)
            rec_counter = 0
            rec_flg = False
            depth_datas=[]
            color_datas=[]
            # 録画一覧更新
            renew_projects()


        # 再生
        if info["play"]:
            if info["past_name"]!=info["project"]: # 初回
                info["past_name"]=info["project"]
                try:
                    depth_datas = np.load(f'save/{info["project"]}/depth_save.npy')
                    color_datas = np.load(f'save/{info["project"]}/color_save.npy')
                    with open(f'save/{info["project"]}/video_info.csv', 'r') as f:
                        reader = csv.reader(f)
                        info_nums = [row for row in reader][0]
                    fps = float(info_nums[1])
                    if info["depth_bool"]: # realsense接続時
                        clipping_distance_in_meters = float(info["meter"])
                        depth_scale = float(info_nums[0])
                        clipping_distance = clipping_distance_in_meters / depth_scale

                except: # ファイルなし
                    depth_datas = []
                    color_datas = []
                    eel.play_switch()
                    continue

            try:
                color_img = color_datas[idx]
                if info["depth_bool"]:
                    depth_img = depth_datas[idx]
                    frame = patch_mask(depth_img, color_img, clipping_distance)
                origin = color_img
                idx+=1
            except IndexError: # 動画終了（IndexError）
                idx = 0
                eel.play_switch()
                continue


        # リアルタイム深度
        elif info["depth_bool"] and not info["play"]:
            clipping_distance_in_meters = float(info["meter"])
            clipping_distance = clipping_distance_in_meters / depth_scale
            frame, origin = conv_depth()
            frame = patch_mask(frame, origin, clipping_distance)
        else:
            ret, origin = cap.read()

        frames = []
        # 無し
        if info["algs"][0]:
            frames.append(origin)
        # 深度
        if info["algs"][1] and info["depth_bool"]:
            frames.append(frame)

        # 背景差分
        for i in range(len(bgs_list)):
            if info["algs"][2:][i]:
                origin_image = copy.deepcopy(origin)
                debug_image = conv_backsub(origin_image, bgs_list[i], info["kernel"])
                frames.append(debug_image)

        concat_list = []
        for i, frame in enumerate(frames):
            if info["tools"][0]: # mediapipe
                debug_image = copy.deepcopy(frame)
                origin_image = copy.deepcopy(origin)
                keypoints = media(debug_image, i)
                if keypoints==None:
                    keypoints=np.zeros((17,2), dtype=int)
                debug_image = draw_debug(
                    debug_image,
                    keypoints,
                )
                concat_list.append(debug_image)
            
            if info["tools"][1]: # movenet
                debug_image = copy.deepcopy(frame)
                origin_image = copy.deepcopy(origin)
                keypoints, scores = run_inference(
                    move_models[i],
                    input_size,
                    frame,
                )
                if keypoints==None:
                    keypoints=np.zeros((17,2), dtype=int)
                debug_image = draw_debug(
                    debug_image,
                    keypoints,
                )
                concat_list.append(debug_image)

            if not info["tools"][0] and not info["tools"][1]:
                concat_list.append(frame)

        if len(concat_list)==0:
            frames = np.zeros((1,1))
        else:
            if len(concat_list)%3==1:
                concat_list.append(np.full((480,640,3), 255))
                concat_list.append(np.full((480,640,3), 255))
            elif len(concat_list)%3==2:
                concat_list.append(np.full((480,640,3), 255))

            rows=[]
            items=[]
            for i, item in enumerate(concat_list):
                items.append(item)
                if (i+1)%3==0:
                    rows.append(np.hstack(items))
                    items = []
            frames = np.vstack(rows)


        _, imencode_image = cv2.imencode('.jpg', frames)
        base64_image = base64.b64encode(imencode_image)
        eel.set_base64image("data:image/jpg;base64," + base64_image.decode("ascii"))
        info["f_counter"]+=1
        eel.set_fps(round(info["f_counter"]/(time.time()-info["start"]), 3))

if __name__ == '__main__':
    main(info, depth_scale)

