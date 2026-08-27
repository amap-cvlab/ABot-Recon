import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from interfaces import ttt3r

infer_videodepth = ttt3r.infer_videodepth
infer_mv_pointclouds = ttt3r.infer_mv_pointclouds
infer_cameras_c2w = ttt3r.infer_cameras_c2w
infer_cameras_w2c = ttt3r.infer_cameras_w2c
infer_monodepth = ttt3r.infer_monodepth