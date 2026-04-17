"""Model factory with explicit baseline dispatch.

- `yolop_baseline.py`   : honest YOLOP-style baseline (DA removed, nc=5).
- `yolopv2_baseline.py` : YOLOPv2-style reconstruction (ELAN + SPPCSPC +
                          transpose-conv decoder, marked [INFERRED]).

`get_net(cfg)` dispatches on `cfg.MODEL.NAME`:
  - 'YOLOP'   -> yolop_baseline.get_net
  - 'YOLOPv2' -> yolopv2_baseline.get_net_yolopv2
"""

from .yolop_baseline import get_net as _get_net_yolop
from .yolopv2_baseline import get_net_yolopv2 as _get_net_yolopv2


def get_net(cfg=None, **kwargs):
    name = 'YOLOPv2'
    if cfg is not None:
        name = getattr(getattr(cfg, 'MODEL', object()), 'NAME', 'YOLOPv2') or 'YOLOPv2'
    name = str(name).strip()
    if name.upper() in ('YOLOP', 'YOLOP-VEHICLE-LANE', 'YOLOP_BASELINE'):
        return _get_net_yolop(cfg, **kwargs)
    if name.upper() in ('YOLOPV2', 'YOLOPV2-VEHICLE-LANE', 'YOLOPV2_BASELINE', 'VEHICLELANE'):
        return _get_net_yolopv2(cfg, **kwargs)
    raise ValueError(
        f"Unknown MODEL.NAME={name!r}. Use 'YOLOP' or 'YOLOPv2'."
    )


# Kept as explicit named exports so notebooks can pick a baseline without
# touching config if they want to.
get_net_yolop = _get_net_yolop
get_net_yolopv2 = _get_net_yolopv2
