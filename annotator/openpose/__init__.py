import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand
from .face import Face
from annotator.util import annotator_ckpts_path


body_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
hand_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
face_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"

class OpenposeDetector:
    def __init__(self):
        body_modelpath = os.path.join(annotator_ckpts_path, "body_pose_model.pth")
        hand_modelpath = os.path.join(annotator_ckpts_path, "hand_pose_model.pth")
        face_modelpath = os.path.join(annotator_ckpts_path, "facenet.pth")

        if not os.path.exists(body_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(body_model_path, model_dir=annotator_ckpts_path)

        if not os.path.exists(hand_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(hand_model_path, model_dir=annotator_ckpts_path)

        if not os.path.exists(face_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(face_model_path, model_dir=annotator_ckpts_path)

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)
        self.face_estimation = Face(face_modelpath)

    def draw_pose(self, pose, H, W):
        bodies = pose['bodies']
        faces = pose['faces']
        hands = pose['hands']
        candidate = bodies['candidate']
        subset = bodies['subset']
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        canvas = util.draw_handpose(canvas, hands)
        canvas = util.draw_facepose(canvas, faces)
        return canvas

    def __call__(self, oriImg, hand_and_face=False, return_is_index=False):
        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            hands = []
            faces = []
            if hand_and_face:
                # Hand
                hands_list = util.handDetect(candidate, subset, oriImg)
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :])
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                        peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                        hands.append(peaks.tolist())
                # Face
                faces_list = util.faceDetect(candidate, subset, oriImg)
                for x, y, w in faces_list:
                    heatmaps = self.face_estimation(oriImg[y:y+w, x:x+w, :])
                    peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                        peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                        faces.append(peaks.tolist())
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            if return_is_index:
                return pose
            else:
                return self.draw_pose(pose, H, W)
