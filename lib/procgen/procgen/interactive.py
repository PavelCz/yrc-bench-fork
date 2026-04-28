#!/usr/bin/env python
import argparse

from procgen import ProcgenGym3Env
from .env import ENV_NAMES
from gym3 import Interactive, VideoRecorderWrapper, unwrap

class ProcgenInteractive(Interactive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_state = None
        self._seeds_exhausted_shown = False

    def _update(self, dt, keys_clicked, keys_pressed):
        # Debug: print info keys once
        if not hasattr(self, "_debug_info_printed"):
            info = self._env.get_info()[0]
            print(f"DEBUG: info keys = {list(info.keys())}")
            self._debug_info_printed = True
        
        if "LEFT_SHIFT" in keys_pressed and "F1" in keys_clicked:
            print("save state")
            self._saved_state = unwrap(self._env).get_state()
        elif "F1" in keys_clicked:
            print("load state")
            if self._saved_state is not None:
                unwrap(self._env).set_state(self._saved_state)
        
        # Check and display if seeds are exhausted
        info = self._env.get_info()[0]
        if "seeds_exhausted" in info:
            val = info["seeds_exhausted"]
            if val and not self._seeds_exhausted_shown:
                print(f"DEBUG: seeds_exhausted={val}, triggering message")
                print("=" * 50)
                print("All level seeds exhausted! Environment is frozen.")
                print("=" * 50)
                self._seeds_exhausted_shown = True
        
        super()._update(dt, keys_clicked, keys_pressed)


def make_interactive(vision, record_dir, **kwargs):
    info_key = None
    ob_key = None
    if vision == "human":
        info_key = "rgb"
        kwargs["render_mode"] = "rgb_array"
    else:
        ob_key = "rgb"

    env = ProcgenGym3Env(num=1, **kwargs)
    
    if record_dir is not None:
        env = VideoRecorderWrapper(
            env=env, directory=record_dir, ob_key=ob_key, info_key=info_key
        )
    h, w, _ = env.ob_space["rgb"].shape
    return ProcgenInteractive(
        env,
        ob_key=ob_key,
        info_key=info_key,
        width=w * 12,
        height=h * 12,
        # keys_to_act=keys_to_act_fn
    )


def main():
    default_str = "(default: %(default)s)"
    parser = argparse.ArgumentParser(
        description="Interactive version of Procgen allowing you to play the games"
    )
    parser.add_argument(
        "--vision",
        default="human",
        choices=["agent", "human"],
        help="level of fidelity of observation " + default_str,
    )
    parser.add_argument("--record-dir", help="directory to record movies to")
    parser.add_argument(
        "--distribution-mode",
        default="hard",
        help="which distribution mode to use for the level generation " + default_str,
    )
    parser.add_argument(
        "--env-name",
        default="heist_aisc_many_chests",
        help="name of game to create " + default_str,
        choices=ENV_NAMES + ["coinrun_old"],
    )
    parser.add_argument(
        "--level-seed", type=int, help="select an individual level to use"
    )
    parser.add_argument(
        "--level-seeds",
        type=int,
        nargs="+",
        help="ordered list of level seeds to use (e.g., --level-seeds 100 200 300)",
    )
    parser.add_argument(
        "--level-seeds-mode",
        default="sequential",
        choices=["sequential", "container", "random"],
        help="how to use level seeds: 'sequential' (in order, stop when exhausted), 'container' (random draw, refill when empty), or 'random' (sample with replacement) " + default_str,
    )

    advanced_group = parser.add_argument_group("advanced optional switch arguments")
    advanced_group.add_argument(
        "--rand-region",
        default=0,
        type=int,
        help="Size of area to randomize cheese location over",
    )
    advanced_group.add_argument(
        "--random-percent",
        default=0,
        type=int,
        help="How often to randomize the level construction",
    )
    advanced_group.add_argument(
        "--key-penalty",
        default=0,
        type=int,
        help="Penalty for picking up keys (divided by 10)",
    )
    advanced_group.add_argument(
        "--step-penalty",
        default=0,
        type=int,
        help="Time penalty per step (divided by 1000)",
    )
    advanced_group.add_argument(
        "--continue-after-coin",
        action="store_true",
        help="If true, don't end the level when coin is collected",
    )
    advanced_group.add_argument(
        "--paint-vel-info",
        action="store_true",
        default=False,
        help="paint player velocity info in the top left corner",
    )
    advanced_group.add_argument(
        "--use-generated-assets",
        action="store_true",
        default=False,
        help="use randomly generated assets in place of human designed assets",
    )
    advanced_group.add_argument(
        "--uncenter-agent",
        action="store_true",
        default=False,
        help="display the full level for games that center the observation to the agent",
    )
    advanced_group.add_argument(
        "--disable-backgrounds",
        action="store_true",
        default=False,
        help="disable human designed backgrounds",
    )
    advanced_group.add_argument(
        "--restrict-themes",
        action="store_true",
        default=False,
        help="restricts games that use multiple themes to use a single theme",
    )
    advanced_group.add_argument(
        "--use-monochrome-assets",
        action="store_true",
        default=False,
        help="use monochromatic rectangles instead of human designed assets",
    )

    args = parser.parse_args()
    
    # Validate level seed arguments
    if args.level_seed is not None and args.level_seeds is not None:
        parser.error("--level-seed and --level-seeds are mutually exclusive")

    kwargs = {
        "paint_vel_info": args.paint_vel_info,
        "use_generated_assets": args.use_generated_assets,
        "center_agent": not args.uncenter_agent,
        "use_backgrounds": not args.disable_backgrounds,
        "restrict_themes": args.restrict_themes,
        "use_monochrome_assets": args.use_monochrome_assets,
        "random_percent": args.random_percent,
        "rand_region": args.rand_region,
        "key_penalty": args.key_penalty,
        "step_penalty": args.step_penalty,
        "continue_after_coin": args.continue_after_coin,
    }
    
    if args.env_name != "coinrun_old":
        kwargs["distribution_mode"] = args.distribution_mode
    if args.level_seed is not None:
        kwargs["start_level"] = args.level_seed
        kwargs["num_levels"] = 1
    if args.level_seeds is not None:
        kwargs["level_seeds"] = args.level_seeds
        kwargs["level_seeds_mode"] = args.level_seeds_mode
        print(f"Using seed list: {args.level_seeds} (mode: {args.level_seeds_mode})")
    ia = make_interactive(
        args.vision, 
        record_dir=args.record_dir, 
        env_name=args.env_name, 
        **kwargs
    )
    
    ia.run()


if __name__ == "__main__":
    main()
