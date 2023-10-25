import datetime
import wandb

class wandb_logger(object):
    def __init__(self, args: dict):
        self.args = args
        self.wandb = None
        if self.args["wandb"]:
            d = datetime.datetime.today()
            self.prefix = f"{d.month}{d.day}{d.hour}{d.minute}{d.second}-{args['seed']}"
            self.wandb = self._create_wandb(args=args)

    def _create_wandb(self, args: dict):
        wandb.login()
        args["prefix"] = self.prefix
        wandb.init(settings=dict(start_method='thread'), project='ad-eval', name=self.prefix, group=args["group_name"], dir="/tmp/wandb")
        wandb.config.update(args)
        return wandb

    def add_image(self, images: list, step: int, log_string):
        if self.args["wandb"]:
            self.wandb.log({log_string: [self.wandb.Image(image) for image in images]}, step=step)

    def add_video(self, path_to_video: str, step: int, log_string, format="gif"):
        if self.args["wandb"]:
            self.wandb.log({log_string: self.wandb.Video(path_to_video, fps=4, format=format)}, step=step)

    def wandb_log(self, data: dict, step: int):
        if self.args["wandb"]:
            self.wandb.log(data, step=step)

    def add_scalar(self, name: str, value: float, step: int):
        if self.args["wandb"]:
            self.wandb.log({name: value}, step=step)
