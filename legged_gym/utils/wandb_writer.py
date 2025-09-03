# utils/wb_writer.py
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter as _TBWriter

# 尝试导入 wandb（没有也不报错，退化为纯 TensorBoard）
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _is_rank0():
    """多卡环境只在 rank0 记录到 wandb。"""
    for k in ("RANK", "SLURM_PROCID", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        v = os.getenv(k)
        if v is not None:
            try:
                return int(v) == 0
            except ValueError:
                return True
    return True


class WandbSummaryWriter(_TBWriter):
    """
    与 torch.utils.tensorboard.SummaryWriter 兼容，并镜像日志到 Weights & Biases。
    关键策略：
      - 所有日志（包括 /time 曲线）都 **显式** 携带一个全局单调递增的 step，避免 W&B 自动自增导致的回退告警
      - 以 "/time" 结尾的曲线，额外携带 'time_sec' 作为横轴（float）
    """

    def __init__(
        self,
        log_dir=None,
        use_wandb=True,
        project="unitree_rl_project",
        run_name=None,
        config=None,
        mode="online",   # "online" | "offline" | "disabled"
        entity=None,
        **tb_kwargs,
    ):
        super().__init__(log_dir=log_dir, **tb_kwargs)

        self._rank0 = _is_rank0()
        self._use_wandb = bool(use_wandb and _WANDB_AVAILABLE and self._rank0)
        self._wb = None
        # 维护一个全局单调递增步计数（只用于 wandb 的 step 避免倒退）
        self._global_step = -1

        if self._use_wandb:
            if wandb.run is None:  # 避免重复 init
                self._wb = wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                    mode=mode,
                    entity=entity,
                    dir=self.get_logdir(),  # 把本地日志和 wandb 存档目录对齐
                )
            else:
                self._wb = wandb.run

            # 记录 tensorboard 日志目录
            try:
                wandb.config.update({"tensorboard_logdir": self.get_logdir()}, allow_val_change=True)
            except Exception:
                pass

            # 定义度量规则：
            #  - 以 /time 结尾的曲线使用 time_sec 为横轴
            #  - 其它曲线使用全局 step（我们在每次 log 时都会提供 step）
            try:
                wandb.define_metric("**/time", step_metric="time_sec")
                # 其余 "**" 无需 define_metric；我们始终传 step 即可
            except Exception:
                pass

    @staticmethod
    def _as_int_or_none(x):
        """尝试把 step 转为 int；失败返回 None。"""
        if x is None:
            return None
        if hasattr(x, "item"):
            x = x.item()
        try:
            return int(x)
        except Exception:
            return None

    def _next_step(self, suggested_step):
        """
        计算这次要传给 wandb 的全局 step：
          - 若调用方给了 suggested_step（通常是 it），优先采用；
            但如果它 <= 当前全局步，则强制 bump 到 current+1，保证单调
          - 若未给，则直接在当前全局步基础上 +1
        """
        s = self._as_int_or_none(suggested_step)
        if s is None:
            self._global_step += 1
        else:
            if s <= self._global_step:
                self._global_step += 1
            else:
                self._global_step = s
        return self._global_step

    # ---------- mirrors ----------

    def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
        # 先写入 TensorBoard
        super().add_scalar(tag, scalar_value, global_step=global_step, **kwargs)

        if not self._use_wandb:
            return

        val = float(_to_numpy(scalar_value))
        step = self._next_step(global_step)  # ☆ 始终显式提供一个单调递增的 step

        if isinstance(tag, str) and tag.endswith("/time"):
            ts = float(global_step) if global_step is not None else None
            self._wb.log({tag: val, "time_sec": ts}, step=step)
        else:
            self._wb.log({tag: val}, step=step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, **kwargs):
        super().add_scalars(main_tag, tag_scalar_dict, global_step=global_step, **kwargs)

        if not self._use_wandb:
            return

        payload = {f"{main_tag}/{k}": float(_to_numpy(v)) for k, v in tag_scalar_dict.items()}
        step = self._next_step(global_step)  # ☆

        if isinstance(main_tag, str) and main_tag.endswith("/time"):
            ts = float(global_step) if global_step is not None else None
            payload["time_sec"] = ts
            self._wb.log(payload, step=step)
        else:
            self._wb.log(payload, step=step)

    def add_image(self, tag, img_tensor, global_step=None, dataformats="CHW", **kwargs):
        super().add_image(tag, img_tensor, global_step=global_step, dataformats=dataformats, **kwargs)

        if self._use_wandb:
            step = self._next_step(global_step)  # ☆
            self._wb.log({tag: wandb.Image(_to_numpy(img_tensor), caption=tag)}, step=step)

    def add_images(self, tag, img_tensor, global_step=None, dataformats="NCHW", **kwargs):
        super().add_images(tag, img_tensor, global_step=global_step, dataformats=dataformats, **kwargs)

        if self._use_wandb:
            step = self._next_step(global_step)  # ☆
            arr = _to_numpy(img_tensor)
            imgs = [wandb.Image(arr[i], caption=f"{tag}/{i}") for i in range(min(len(arr), 16))]
            self._wb.log({tag: imgs}, step=step)

    def add_histogram(self, tag, values, global_step=None, **kwargs):
        super().add_histogram(tag, values, global_step=global_step, **kwargs)

        if self._use_wandb:
            step = self._next_step(global_step)  # ☆
            self._wb.log({tag: wandb.Histogram(_to_numpy(values))}, step=step)

    def add_text(self, tag, text_string, global_step=None, **kwargs):
        super().add_text(tag, text_string, global_step=global_step, **kwargs)

        if self._use_wandb:
            step = self._next_step(global_step)  # ☆
            self._wb.log({tag: str(text_string)}, step=step)

    def add_figure(self, tag, figure, global_step=None, **kwargs):
        super().add_figure(tag, figure, global_step=global_step, **kwargs)

        if self._use_wandb:
            step = self._next_step(global_step)  # ☆
            if isinstance(figure, list):
                self._wb.log({tag: [wandb.Image(f) for f in figure]}, step=step)
            else:
                self._wb.log({tag: wandb.Image(figure)}, step=step)

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, **kwargs):
        super().add_video(tag, vid_tensor, global_step=global_step, fps=fps, **kwargs)

        if self._use_wandb:
            step = self._next_step(global_step)  # ☆
            arr = _to_numpy(vid_tensor)  # (N, T, C, H, W)
            clip = arr[0]  # 截第一段，避免日志过大
            self._wb.log({tag: wandb.Video(clip, fps=fps, format="mp4")}, step=step)

    def add_hparams(self, hparam_dict, metric_dict, run_name=None, global_step=None, **kwargs):
        super().add_hparams(hparam_dict, metric_dict, run_name=run_name, global_step=global_step, **kwargs)

        if self._use_wandb:
            try:
                wandb.config.update(hparam_dict, allow_val_change=True)
            except Exception:
                pass
            step = self._next_step(global_step)  # ☆
            flat = {f"hparam_metric/{k}": float(_to_numpy(v)) for k, v in metric_dict.items()}
            self._wb.log(flat, step=step)

    # ---------- lifecycle ----------

    def flush(self):
        super().flush()

    def close(self):
        super().close()
        if self._use_wandb and self._wb is not None:
            try:
                wandb.finish()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
