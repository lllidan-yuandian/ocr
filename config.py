

from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.train = edict()
__C.anchor_scales=[8.0, 16.0, 32.0]
__C.anchor_ratios=[0.5, 1, 2]

__C.train.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.train.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.train.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.train.RPN_MIN_SIZE = 8
__C.FEAT_STRIDE=[16, ]
