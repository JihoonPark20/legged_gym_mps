# SPDX-License-Identifier: BSD-3-Clause
#
# W&B Parameter Sweep entry point for legged_gym.
#
# 학습 config(LeggedRobotCfg / LeggedRobotCfgPPO)에 정의된 하이퍼파라미터를
# Weights & Biases 의 Sweep 기능으로 자동 탐색합니다.
#
# 동작 방식
# ---------
# 1. sweep_config.yaml 을 읽어 W&B sweep 을 생성(또는 기존 sweep 에 연결)합니다.
# 2. wandb.agent 가 매 run 마다 wandb.config(샘플링된 하이퍼파라미터)를 제공합니다.
# 3. train_for_sweep() 이 그 값을 env_cfg / train_cfg 에 주입한 뒤 학습을 실행합니다.
# 4. wandb.init(sync_tensorboard=True) 로 rsl_rl 의 TensorBoard 스칼라를 자동 미러링하여
#    metric("Train/mean_reward")을 W&B 가 추적/최적화합니다.
#
# 사용 예시
# ---------
#   # 새 sweep 생성 + agent 1개 실행 (500 iters/run 권장)
#   python legged_gym/scripts/sweep.py --task go2 --headless --max_iterations 500
#
#   # 다른 터미널/머신에서 같은 sweep 에 agent 추가 (병렬 탐색)
#   python legged_gym/scripts/sweep.py --task go2 --headless --max_iterations 500 \
#       --sweep_id <entity>/<project>/<sweep_id> --sweep_count 5
#
import os

import wandb
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403  (registers tasks in task_registry)
from legged_gym.utils import get_args, task_registry

# train_cfg(LeggedRobotCfgPPO) 로 라우팅되는 최상위 그룹.
# 그 외의 점-경로(rewards.*, control.*, env.* 등)는 env_cfg 로 적용됩니다.
_TRAIN_CFG_GROUPS = {"algorithm", "policy", "runner"}


def _set_nested_attr(obj, dotted_key, value):
    """ 'algorithm.learning_rate' 같은 점-경로를 따라가 마지막 속성을 설정한다. """
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(
            f"Sweep parameter '{dotted_key}' not found in config "
            f"(missing attribute '{parts[-1]}'). YAML 의 키 경로를 확인하세요."
        )
    setattr(obj, parts[-1], value)


def apply_sweep_config(sweep_params, env_cfg, train_cfg):
    """ wandb.config(평탄한 점-경로 dict)를 env_cfg/train_cfg 에 주입한다. """
    applied = {}
    for key, value in sweep_params.items():
        top = key.split(".")[0]
        if top in _TRAIN_CFG_GROUPS or key == "seed":
            target = train_cfg
        else:
            target = env_cfg
        _set_nested_attr(target, key, value)
        applied[key] = value
    print("[sweep] applied hyperparameters:")
    for key, value in applied.items():
        print(f"    {key} = {value}")
    return applied


def train_for_sweep(args):
    """ wandb.agent 가 매 run 마다 호출하는 학습 함수. """
    # sync_tensorboard=True: rsl_rl 의 SummaryWriter 스칼라를 W&B 로 자동 미러링.
    # (러너가 SummaryWriter 를 만들기 전에 init 되어야 하므로 학습 시작 전에 호출)
    run = wandb.init(sync_tensorboard=True)
    try:
        # 1) 기본 config 로드 (등록된 task 기준)
        env_cfg, train_cfg = task_registry.get_cfgs(args.task)

        # 2) sweep 이 샘플링한 하이퍼파라미터 주입
        apply_sweep_config(dict(wandb.config), env_cfg, train_cfg)

        # 3) run 식별용 이름/실험명 설정 (로그 디렉터리 정리에 유용)
        train_cfg.runner.run_name = run.name or run.id

        # 4) 환경 + 러너 생성 (override 한 cfg 전달)
        env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=None, args=args, train_cfg=train_cfg
        )

        # 5) 학습 실행 (max_iterations 는 --max_iterations 로 덮어쓰기 가능)
        runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        run.finish()


def _load_sweep_config(args):
    path = args.sweep_config or os.path.join(
        LEGGED_GYM_ROOT_DIR, "legged_gym", "scripts", "sweep_config.yaml"
    )
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[sweep] loaded sweep config from: {path}")
    return config


def main():
    args = get_args()

    # 기존 sweep 에 agent 만 붙일지, 새 sweep 을 만들지 결정
    if args.sweep_id is not None:
        sweep_id = args.sweep_id
        print(f"[sweep] attaching agent to existing sweep: {sweep_id}")
    else:
        sweep_config = _load_sweep_config(args)
        sweep_id = wandb.sweep(
            sweep_config, project=args.wandb_project, entity=args.wandb_entity
        )
        print(f"[sweep] created new sweep: {sweep_id}")
        print(f"[sweep] 다른 머신에서 병렬 실행하려면: --sweep_id {sweep_id}")

    # agent 실행: count 만큼 run 을 수행
    wandb.agent(
        sweep_id,
        function=lambda: train_for_sweep(args),
        count=args.sweep_count,
        project=args.wandb_project,
        entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()
