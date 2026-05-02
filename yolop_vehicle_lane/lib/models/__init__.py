from .yolop_baseline import get_net as get_net_yolop
from .yolopv2_baseline import get_net_yolopv2
from .yolopx_baseline import get_net_yolopx


def get_net(cfg, **kwargs):
    name = str(getattr(cfg.MODEL, 'NAME', 'YOLOP')).lower()
    if name == 'yolop':
        return get_net_yolop(cfg, **kwargs)
    if name == 'yolopv2':
        return get_net_yolopv2(cfg, **kwargs)
    if name == 'yolopx':
        return get_net_yolopx(cfg, **kwargs)
    raise ValueError(f'Unsupported MODEL.NAME={cfg.MODEL.NAME!r}. Expected YOLOP, YOLOPv2, or YOLOPX.')
